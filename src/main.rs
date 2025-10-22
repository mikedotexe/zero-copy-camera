use metal::*;
use minifb::{Key, Window, WindowOptions};
use nokhwa::{
    query,
    utils::{ApiBackend, RequestedFormat, RequestedFormatType},
    Camera, pixel_format::RgbFormat,
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::{
    slice,
    sync::{Arc, atomic::{AtomicUsize, Ordering}, Mutex},
};

const W: usize = 1280;
const H: usize = 720;

const SRC: &str = include_str!("../shaders/camera_effects.metal");

#[repr(C)]
struct Params {
    time: f32,
    w: u32,
    h: u32,
    wind_strength: f32,
    warp_pattern: f32,
    audio_bass: f32,
    audio_mid: f32,
    audio_treble: f32,
    show_grid: u32,
    camera_mode: u32,
}

// Audio ring buffer (copied from warped_plane_demo)
struct AudioRingBuffer {
    data: Vec<f32>,
    write_pos: Arc<AtomicUsize>,
}

impl AudioRingBuffer {
    fn new(size: usize) -> Self {
        Self {
            data: vec![0.0; size],
            write_pos: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn get_samples(&self, out: &mut [f32]) {
        let data_len = self.data.len();
        let out_len = out.len();
        let mask = data_len - 1;
        let pos = self.write_pos.load(Ordering::Relaxed);

        for (i, sample) in out.iter_mut().enumerate() {
            let idx = (pos + data_len - out_len + i) & mask;
            *sample = self.data[idx];
        }
    }
}

fn analyze_spectrum(audio_data: &[f32], spectrum: &mut [f32; 3]) {
    const AUDIO_GAIN: f32 = 12.0;
    let chunk_size = audio_data.len() / 3;

    for i in 0..3 {
        let start = i * chunk_size;
        let end = ((i + 1) * chunk_size).min(audio_data.len());

        let mut sum = 0.0f32;
        for &sample in &audio_data[start..end] {
            sum += sample * sample;
        }

        let rms = (sum / (end - start) as f32).sqrt();
        let amplified = (rms * AUDIO_GAIN).min(1.0);
        spectrum[i] = spectrum[i] * 0.7 + amplified * 0.3;
        spectrum[i] = spectrum[i].clamp(0.0, 1.0);
    }
}

fn setup_audio_capture(
    audio_ring: Arc<Mutex<AudioRingBuffer>>,
) -> Result<cpal::Stream, Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let device = host.default_input_device()
        .ok_or("No default input device")?;

    let supported_config = device.default_input_config()?;
    let config = supported_config.config();

    let err_fn = |err| eprintln!("Audio stream error: {}", err);

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| {
            let mut ring = audio_ring.lock().unwrap();
            let data_len = ring.data.len();
            let mask = data_len - 1;
            let start = ring.write_pos.load(Ordering::Relaxed);

            for (i, &sample) in data.iter().enumerate() {
                let idx = (start + i) & mask;
                ring.data[idx] = sample;
            }

            ring.write_pos.store((start + data.len()) & mask, Ordering::Relaxed);
        },
        err_fn,
        None,
    )?;

    stream.play()?;
    Ok(stream)
}

fn pso(device: &Device, entry: &str) -> ComputePipelineState {
    let lib = device.new_library_with_source(SRC, &CompileOptions::new())
        .unwrap_or_else(|e| panic!("Failed to compile shader: {}", e));
    let f = lib.get_function(entry, None)
        .expect(&format!("Failed to find function '{}'", entry));
    device.new_compute_pipeline_state_with_function(&f)
        .unwrap_or_else(|e| panic!("Failed to create pipeline: {}", e))
}

fn main() {
    println!("🌊 Two Sheets - Artistic Camera Effects");
    println!("========================================");
    println!("Two camera sheets with artistic warp patterns & audio-reactive strings");
    println!();
    println!("Camera Modes (keys 1-3):");
    println!("  1 - Centered & Flipped (mirror)");
    println!("  2 - Zoomed Out, No Flip");
    println!("  3 - Tight Crop, Flipped");
    println!();
    println!("Warp Patterns (keys 4-8):");
    println!("  4 - Sine Wave Ripples");
    println!("  5 - Spiral Vortex");
    println!("  6 - Spherical Bulge");
    println!("  7 - Checkerboard Twist");
    println!("  8 - Hyperbolic Flow");
    println!();
    println!("Other Keys:");
    println!("  G - Grid Toggle");
    println!("  A - Audio Toggle");
    println!("  UP/DOWN - Wind Strength");
    println!("  R - Reset");
    println!("  Esc - Quit");
    println!();

    let mut window = Window::new(
        "Two Sheets - Artistic Camera Effects",
        W, H, WindowOptions::default(),
    ).unwrap();
    window.limit_update_rate(Some(std::time::Duration::from_micros(16_600)));

    // Audio setup
    let audio_ring = Arc::new(Mutex::new(AudioRingBuffer::new(8192)));
    let _audio_stream = setup_audio_capture(audio_ring.clone())
        .unwrap_or_else(|e| {
            eprintln!("⚠️  Audio failed: {}. Continuing without audio.", e);
            setup_audio_capture(Arc::new(Mutex::new(AudioRingBuffer::new(1024)))).unwrap()
        });
    let mut audio_samples = vec![0.0f32; 2048];
    let mut spectrum = [0.0f32; 3];

    let device = Device::system_default().expect("no Metal device");
    let q = device.new_command_queue();

    let p_render = pso(&device, "render_two_sheets");

    // Camera setup
    let cams = query(ApiBackend::Auto).expect("camera query");
    let info = cams.first().expect("no camera");

    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut cam = Camera::new(info.index().clone(), format).expect("camera open");
    cam.open_stream().expect("open stream");

    let frame = cam.frame().expect("camera frame");
    let cam_w = frame.resolution().width() as usize;
    let cam_h = frame.resolution().height() as usize;
    println!("Camera: {}×{}", cam_w, cam_h);

    // Buffers
    let cam_buf = device.new_buffer((W * H * 4) as u64, MTLResourceOptions::StorageModeShared);
    let out_buf = device.new_buffer((W * H * 4) as u64, MTLResourceOptions::StorageModeShared);
    let params_buf = device.new_buffer(std::mem::size_of::<Params>() as u64, MTLResourceOptions::StorageModeShared);

    let mut time = 0.0f32;
    let mut wind_strength = 1.0f32;
    let mut warp_pattern = 0.0f32;
    let mut show_grid = 0u32;
    let mut audio_enabled = true;
    let mut camera_mode = 0u32;

    println!("Starting...\n");
    println!("🌊 Pattern: Sine Wave Ripples");
    println!("📷 Camera Mode 0: Centered & Flipped");

    loop {
        if !window.is_open() || window.is_key_down(Key::Escape) {
            break;
        }

        // Camera mode selection (keys 1-3)
        if window.is_key_pressed(Key::Key1, minifb::KeyRepeat::No) {
            camera_mode = 0;
            println!("📷 Camera Mode 0: Centered & Flipped");
        }
        if window.is_key_pressed(Key::Key2, minifb::KeyRepeat::No) {
            camera_mode = 1;
            println!("📷 Camera Mode 1: Zoomed Out, No Flip");
        }
        if window.is_key_pressed(Key::Key3, minifb::KeyRepeat::No) {
            camera_mode = 2;
            println!("📷 Camera Mode 2: Tight Crop, Flipped");
        }

        // Warp patterns (keys 4-8)
        if window.is_key_pressed(Key::Key4, minifb::KeyRepeat::No) {
            warp_pattern = 0.0;
            println!("🌊 Warp: Sine Wave Ripples");
        }
        if window.is_key_pressed(Key::Key5, minifb::KeyRepeat::No) {
            warp_pattern = 1.0;
            println!("🌀 Warp: Spiral Vortex");
        }
        if window.is_key_pressed(Key::Key6, minifb::KeyRepeat::No) {
            warp_pattern = 2.0;
            println!("🔮 Warp: Spherical Bulge");
        }
        if window.is_key_pressed(Key::Key7, minifb::KeyRepeat::No) {
            warp_pattern = 3.0;
            println!("♟️  Warp: Checkerboard Twist");
        }
        if window.is_key_pressed(Key::Key8, minifb::KeyRepeat::No) {
            warp_pattern = 4.0;
            println!("💫 Warp: Hyperbolic Flow");
        }

        // Grid toggle
        if window.is_key_pressed(Key::G, minifb::KeyRepeat::No) {
            show_grid = if show_grid == 0 { 1 } else { 0 };
            println!("Grid: {}", if show_grid == 1 { "ON" } else { "OFF" });
        }

        // Audio toggle
        if window.is_key_pressed(Key::A, minifb::KeyRepeat::No) {
            audio_enabled = !audio_enabled;
            println!("Audio: {}", if audio_enabled { "ON" } else { "OFF" });
        }

        // Camera capture with CENTER CROP
        if let Ok(frame) = cam.frame() {
            let rgb = frame.decode_image::<RgbFormat>().expect("decode");
            let rgb_data = rgb.as_raw();

            // Calculate center crop offsets
            // Camera is 1920x1080, output is 1280x720
            let x_offset = (cam_w - W) / 2;  // (1920-1280)/2 = 320 pixels from left
            let y_offset = (cam_h - H) / 2;  // (1080-720)/2 = 180 pixels from top

            unsafe {
                let ptr = cam_buf.contents() as *mut u32;
                for y in 0..H {
                    for x in 0..W {
                        // Sample from center of camera frame
                        let src_x = x + x_offset;
                        let src_y = y + y_offset;
                        let src_idx = (src_y * cam_w + src_x) * 3;

                        let r = rgb_data[src_idx + 0] as u32;
                        let g = rgb_data[src_idx + 1] as u32;
                        let b = rgb_data[src_idx + 2] as u32;
                        let pixel = (r << 16) | (g << 8) | b;
                        ptr.offset((y * W + x) as isize).write(pixel);
                    }
                }
            }
        }

        // Audio analysis
        if audio_enabled {
            if let Ok(ring) = audio_ring.lock() {
                ring.get_samples(&mut audio_samples);
                analyze_spectrum(&audio_samples, &mut spectrum);
            }
        } else {
            spectrum = [0.0; 3];
        }

        // Update parameters
        let params = Params {
            time,
            w: W as u32,
            h: H as u32,
            wind_strength,
            warp_pattern,
            audio_bass: spectrum[0],
            audio_mid: spectrum[1],
            audio_treble: spectrum[2],
            show_grid,
            camera_mode,
        };

        unsafe {
            std::ptr::copy_nonoverlapping(
                &params as *const Params as *const u8,
                params_buf.contents() as *mut u8,
                std::mem::size_of::<Params>(),
            );
        }

        time += 0.016;

        // GPU rendering
        {
            let cmd = q.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();

            enc.set_compute_pipeline_state(&p_render);
            enc.set_buffer(0, Some(&cam_buf), 0);
            enc.set_buffer(1, Some(&out_buf), 0);
            enc.set_buffer(2, Some(&params_buf), 0);

            let tg = MTLSize::new(16, 16, 1);
            let grid_w = ((W + 15) / 16) as u64;
            let grid_h = ((H + 15) / 16) as u64;
            enc.dispatch_thread_groups(MTLSize::new(grid_w, grid_h, 1), tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // Display
        let view = unsafe { slice::from_raw_parts(out_buf.contents() as *const u32, W * H) };
        window.update_with_buffer(view, W, H).unwrap();

        // Controls
        if window.is_key_down(Key::Up) {
            wind_strength = (wind_strength + 0.02).min(3.0);
            println!("Wind: {:.2}", wind_strength);
        }
        if window.is_key_down(Key::Down) {
            wind_strength = (wind_strength - 0.02).max(0.0);
            println!("Wind: {:.2}", wind_strength);
        }
        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            wind_strength = 1.0;
            time = 0.0;
            println!("Reset - Wind: {:.2}", wind_strength);
        }
    }
}
