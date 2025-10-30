use crate::starlit_blossom_713::spiral_portal_burst;
use metal::{Buffer as MTLBuffer, *};
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use nokhwa::{
    pixel_format::RgbFormat,
    query,
    utils::{ApiBackend, RequestedFormat, RequestedFormatType},
    Camera,
};
use std::{
    f32::consts::TAU,
    slice,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

#[cfg(feature = "audio")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

#[cfg(feature = "audio")]
type AudioStream = Option<cpal::Stream>;

#[cfg(not(feature = "audio"))]
type AudioStream = ();

#[cfg(target_os = "macos")]
use cocoa::{
    appkit::{NSColor, NSView, NSWindow, NSWindowStyleMask, NSWindowTitleVisibility},
    base::nil,
    foundation::NSString,
};
#[cfg(target_os = "macos")]
use core_graphics::{
    geometry::{CGAffineTransform, CGPoint, CGRect, CGSize},
    path::CGMutablePath,
};
#[cfg(target_os = "macos")]
use objc::runtime::{Object, NO, YES};
#[cfg(target_os = "macos")]
use objc::{class, msg_send, sel, sel_impl};

const W: usize = 1280;
const H: usize = 720;

const SRC: &str = include_str!("../shaders/camera_effects.metal");

const PATCH_SIDE: usize = 4;
const PATCH_PIXELS: usize = PATCH_SIDE * PATCH_SIDE;
const MLP_INPUTS: usize = PATCH_PIXELS * 3;
const MLP_PROJ: usize = 16;
const MLP_OUTPUTS: usize = 3;
const MAX_HOLES: usize = 8;

#[repr(C)]
#[derive(Clone, Copy)]
struct NeuralWeights {
    proj_bg: [[f32; MLP_INPUTS]; MLP_PROJ],
    proj_sheet: [[f32; MLP_INPUTS]; MLP_PROJ],
    bias_bg: [f32; MLP_PROJ],
    bias_sheet: [f32; MLP_PROJ],
    mix_bg: [[f32; MLP_PROJ]; MLP_OUTPUTS],
    mix_sheet: [[f32; MLP_PROJ]; MLP_OUTPUTS],
    bias_mix_bg: [f32; MLP_OUTPUTS],
    bias_mix_sheet: [f32; MLP_OUTPUTS],
}
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Analysis {
    heavy_ops: u32,
    mlp_energy: u32,
    brightest_activation: u32,
    edge_flux: u32,
}

#[repr(C)]
struct Params {
    time: f32,
    w: u32,
    h: u32,
    cam_w: u32,
    cam_h: u32,
    cam_stride: u32,
    cam_offset_x: u32,
    cam_offset_y: u32,
    cam_channels: u32,
    wind_strength: f32,
    warp_pattern: f32,
    audio_bass: f32,
    audio_mid: f32,
    audio_treble: f32,
    show_grid: u32,
    camera_mode: u32,
    mouse_x: f32,
    mouse_y: f32,
    neural_gain: f32,
    neural_enabled: u32,
    hole_count: u32,
    hole_pad: [u32; 3],
    holes: [HoleParam; MAX_HOLES],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct HoleParam {
    packed: [f32; 4],
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
    let device = host
        .default_input_device()
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

            ring.write_pos
                .store((start + data.len()) & mask, Ordering::Relaxed);
        },
        err_fn,
        None,
    )?;

    stream.play()?;
    Ok(stream)
}

fn pso(device: &Device, entry: &str) -> ComputePipelineState {
    let lib = device
        .new_library_with_source(SRC, &CompileOptions::new())
        .unwrap_or_else(|e| panic!("Failed to compile shader: {}", e));
    let f = lib
        .get_function(entry, None)
        .expect(&format!("Failed to find function '{}'", entry));
    device
        .new_compute_pipeline_state_with_function(&f)
        .unwrap_or_else(|e| panic!("Failed to create pipeline: {}", e))
}

fn init_neural_weights() -> NeuralWeights {
    let mut weights = NeuralWeights {
        proj_bg: [[0.0; MLP_INPUTS]; MLP_PROJ],
        proj_sheet: [[0.0; MLP_INPUTS]; MLP_PROJ],
        bias_bg: [0.0; MLP_PROJ],
        bias_sheet: [0.0; MLP_PROJ],
        mix_bg: [[0.0; MLP_PROJ]; MLP_OUTPUTS],
        mix_sheet: [[0.0; MLP_PROJ]; MLP_OUTPUTS],
        bias_mix_bg: [0.0; MLP_OUTPUTS],
        bias_mix_sheet: [0.0; MLP_OUTPUTS],
    };

    for proj in 0..MLP_PROJ {
        for i in 0..MLP_INPUTS {
            let seed = (proj * MLP_INPUTS + i) as f32 * 0.137 + 0.37;
            let swirl = (seed * 1.3).sin();
            weights.proj_bg[proj][i] = (seed.sin() * 0.7 + seed.cos() * 0.3) * 0.05 + swirl * 0.01;
            let sheet_seed = seed + 1.7;
            weights.proj_sheet[proj][i] =
                (sheet_seed.cos() * 0.6 + sheet_seed.sin() * 0.4) * 0.045 - swirl * 0.008;
        }

        let bias_seed = proj as f32 * 0.271;
        weights.bias_bg[proj] = (bias_seed.sin() + bias_seed.cos() * 0.5) * 0.08;
        weights.bias_sheet[proj] = (bias_seed.cos() - bias_seed.sin() * 0.25) * 0.07;
    }

    for o in 0..MLP_OUTPUTS {
        for proj in 0..MLP_PROJ {
            let seed = (o * MLP_PROJ + proj) as f32 * 0.199 + 0.11 * o as f32;
            weights.mix_bg[o][proj] = (seed.sin() * 0.6 + seed.cos() * 0.4) * 0.09;
            weights.mix_sheet[o][proj] = (seed.cos() * 0.7 + seed.sin() * 0.3) * 0.11;
        }
        let bias_seed = (o as f32 + 1.0) * 0.77;
        weights.bias_mix_bg[o] = bias_seed.sin() * 0.15;
        weights.bias_mix_sheet[o] = bias_seed.cos() * 0.18;
    }

    weights
}

fn retune_weights(weights: &mut NeuralWeights, phase: f32) {
    let sweep = (phase * 0.7).sin();
    let shimmer = (phase * 1.3).cos();

    for proj in 0..MLP_PROJ {
        let lfo = (phase + proj as f32 * 0.19).sin();
        for i in 0..MLP_INPUTS {
            let drive = ((proj + i) as f32 * 0.17 + phase * 0.43).sin();
            let wave = (drive * TAU).sin();
            weights.proj_bg[proj][i] =
                (weights.proj_bg[proj][i] * 0.8) + wave * (0.08 + sweep.abs() * 0.05);
            weights.proj_sheet[proj][i] =
                (weights.proj_sheet[proj][i] * 0.78) + (wave * 0.06 + shimmer * 0.03);
        }

        weights.bias_bg[proj] = weights.bias_bg[proj] * 0.7 + lfo * 0.22;
        weights.bias_sheet[proj] = weights.bias_sheet[proj] * 0.72 + lfo.cos() * 0.2;
    }

    for o in 0..MLP_OUTPUTS {
        let burst = (phase * 0.9 + o as f32 * 0.63).sin();
        for proj in 0..MLP_PROJ {
            let w = (proj as f32 * 0.31 + phase * 0.57).cos();
            weights.mix_bg[o][proj] = (weights.mix_bg[o][proj] * 0.82) + w * (0.07 + burst * 0.03);
            weights.mix_sheet[o][proj] =
                (weights.mix_sheet[o][proj] * 0.8) + (w * 0.09 + shimmer * 0.04);
        }

        weights.bias_mix_bg[o] = weights.bias_mix_bg[o] * 0.68 + burst * 0.21;
        weights.bias_mix_sheet[o] = weights.bias_mix_sheet[o] * 0.7 + burst.cos() * 0.24;
    }
}

fn animate_weights(weights: &mut NeuralWeights, spectrum: &[f32; 3], mouse: (f32, f32), time: f32) {
    let audio_push = spectrum[0] * 0.8 + spectrum[1] * 1.1 + spectrum[2] * 1.4;
    let mouse_energy = (mouse.0 - 0.5).abs() + (mouse.1 - 0.5).abs();

    for proj in 0..MLP_PROJ {
        let drift = (time * 0.35 + proj as f32 * 0.23).sin();
        for i in 0..MLP_INPUTS {
            let base = (i as f32 * 0.021 + proj as f32 * 0.037 + time * 0.42).sin();
            let modulator = base * (0.004 + audio_push * 0.002) + drift * 0.0025;
            let sheet_mod = base.cos() * (0.003 + spectrum[2] * 0.0035);

            weights.proj_bg[proj][i] = (weights.proj_bg[proj][i] + modulator).clamp(-0.42, 0.42);
            weights.proj_sheet[proj][i] =
                (weights.proj_sheet[proj][i] + sheet_mod + modulator * 0.6).clamp(-0.45, 0.45);
        }

        weights.bias_bg[proj] =
            (weights.bias_bg[proj] * 0.995) + drift * 0.012 + audio_push * 0.018;
        weights.bias_sheet[proj] =
            (weights.bias_sheet[proj] * 0.993) + (mouse_energy * 0.03) + drift.cos() * 0.01;
    }

    for o in 0..MLP_OUTPUTS {
        let swirl = (time * 0.27 + o as f32 * 1.1).sin();
        for proj in 0..MLP_PROJ {
            let adjust = swirl * 0.004 + audio_push * 0.003;
            weights.mix_bg[o][proj] = (weights.mix_bg[o][proj] + adjust).clamp(-0.6, 0.6);
            let sheet_adjust = adjust * (1.0 + mouse.0 * 0.6) + spectrum[2] * 0.004;
            weights.mix_sheet[o][proj] =
                (weights.mix_sheet[o][proj] + sheet_adjust).clamp(-0.7, 0.7);
        }

        weights.bias_mix_bg[o] = (weights.bias_mix_bg[o] * 0.997) + audio_push * 0.01;
        weights.bias_mix_sheet[o] =
            (weights.bias_mix_sheet[o] * 0.996) + (mouse.1 - 0.5) * 0.025 + swirl * 0.008;
    }
}

#[derive(Clone)]
struct Hole {
    center: [f32; 2],
    radius_base: f32,
    radius: f32,
    vigor: f32,
    phase: f32,
    edge_side: Option<u8>,
    edge_progress: f32,
    free: bool,
    age: f32,
}

impl Hole {
    fn new_edge(side: u8, progress: f32) -> Self {
        let mut hole = Hole {
            center: [0.5, 0.5],
            radius_base: 0.12,
            radius: 0.12,
            vigor: 0.6,
            phase: progress * TAU,
            edge_side: Some(side),
            edge_progress: progress.fract(),
            free: false,
            age: 0.0,
        };

        hole.center = match side % 4 {
            0 => [hole.edge_progress, 0.06],
            1 => [hole.edge_progress, 0.94],
            2 => [0.06, hole.edge_progress],
            _ => [0.94, hole.edge_progress],
        };

        hole
    }

    fn new_free(cx: f32, cy: f32, radius: f32) -> Self {
        Hole {
            center: [cx.clamp(0.05, 0.95), cy.clamp(0.05, 0.95)],
            radius_base: radius.clamp(0.04, 0.18),
            radius: radius.clamp(0.04, 0.18),
            vigor: 0.5,
            phase: (cx + cy) * TAU,
            edge_side: None,
            edge_progress: 0.0,
            free: true,
            age: 0.0,
        }
    }

    fn to_param(&self) -> HoleParam {
        HoleParam {
            packed: [
                self.center[0],
                self.center[1],
                self.radius.max(0.01),
                self.vigor.max(0.05),
            ],
        }
    }
}

fn seed_default_holes() -> Vec<Hole> {
    let mut holes = Vec::with_capacity(MAX_HOLES);
    for i in 0..4 {
        holes.push(Hole::new_edge(i as u8, (i as f32) / 4.0));
    }
    holes.push(Hole::new_edge(0, 0.42));
    holes.push(Hole::new_edge(1, 0.78));
    holes
}

fn update_hole_field(
    holes: &mut Vec<Hole>,
    dt: f32,
    mouse: (f32, f32),
    flux: f32,
    collapse_factor: f32,
) -> [HoleParam; MAX_HOLES] {
    let energy = 0.25 + flux * 1.5;

    for (index, hole) in holes.iter_mut().enumerate() {
        hole.age += dt;
        hole.phase += dt * (0.7 + hole.vigor * 0.7 + energy * 0.4);
        let wobble = (hole.phase + hole.edge_progress * TAU + index as f32 * 0.37).sin();
        let drift = (hole.phase * 0.5 + flux * 0.3).cos();

        if let Some(side) = hole.edge_side {
            let direction = if side % 2 == 0 { 1.0 } else { -1.0 };
            hole.edge_progress =
                (hole.edge_progress + dt * (0.05 + energy * 0.04) * direction).fract();
            if hole.edge_progress < 0.0 {
                hole.edge_progress += 1.0;
            }
            hole.center = match side % 4 {
                0 => [hole.edge_progress, 0.05 + 0.03 * wobble],
                1 => [hole.edge_progress, 0.95 - 0.03 * wobble],
                2 => [0.05 + 0.03 * wobble, hole.edge_progress],
                _ => [0.95 - 0.03 * wobble, hole.edge_progress],
            };
        } else {
            let accel = 0.32 + energy * 0.12;
            let to_mouse = [mouse.0 - hole.center[0], mouse.1 - hole.center[1]];
            hole.center[0] =
                (hole.center[0] + to_mouse[0] * accel * dt + wobble * 0.012).clamp(0.05, 0.95);
            hole.center[1] =
                (hole.center[1] + to_mouse[1] * accel * dt + drift * 0.012).clamp(0.05, 0.95);
        }

        hole.vigor = 0.35 + energy * 0.75;
        let collapse_mix = 0.55 + 0.45 * collapse_factor;
        let pulse = 0.62 + 0.38 * ((wobble * 0.5 + 0.5) + flux * 0.22);
        let desired = hole.radius_base * collapse_mix * pulse;
        hole.radius = hole.radius * 0.8 + desired * 0.2;
    }

    holes.retain(|h| !h.free || h.age < 40.0);

    while holes.len() > MAX_HOLES {
        if let Some(idx) = holes
            .iter()
            .enumerate()
            .filter(|(_, h)| h.free)
            .min_by(|a, b| a.1.radius.total_cmp(&b.1.radius))
            .map(|(idx, _)| idx)
        {
            holes.remove(idx);
        } else {
            break;
        }
    }

    let mut packed = [HoleParam::default(); MAX_HOLES];
    for (i, hole) in holes.iter().enumerate().take(MAX_HOLES) {
        packed[i] = hole.to_param();
    }
    packed
}

#[cfg(target_os = "macos")]
struct SwissMask {
    content_view: *mut Object,
    mask_layer: *mut Object,
}

#[cfg(target_os = "macos")]
fn setup_window_mask(window: &Window) -> Option<SwissMask> {
    unsafe {
        let ns_window = window.get_window_handle() as *mut Object;
        if ns_window.is_null() {
            return None;
        }

        let style_mask = NSWindowStyleMask::NSWindowStyleMaskBorderless
            | NSWindowStyleMask::NSWindowStyleMaskResizable
            | NSWindowStyleMask::NSWindowStyleMaskFullSizeContentView;
        let _: () = msg_send![ns_window, setStyleMask: style_mask];
        let _: () = msg_send![ns_window, setTitleVisibility: NSWindowTitleVisibility::Hidden];
        let _: () = msg_send![ns_window, setTitlebarAppearsTransparent: YES];
        let _: () = msg_send![ns_window, setMovableByWindowBackground: YES];
        let clear: *mut Object =
            msg_send![class!(NSColor), colorWithCalibratedRed:0.0 green:0.0 blue:0.0 alpha:0.0];
        let _: () = msg_send![ns_window, setBackgroundColor: clear];
        let _: () = msg_send![ns_window, setOpaque: NO];
        let _: () = msg_send![ns_window, setHasShadow: NO];

        let content_view: *mut Object = msg_send![ns_window, contentView];
        if content_view.is_null() {
            return None;
        }

        let _: () = msg_send![content_view, setWantsLayer: YES];
        let layer: *mut Object = msg_send![content_view, layer];
        let _: () = msg_send![layer, setMasksToBounds: NO];

        let mask_layer: *mut Object = msg_send![class!(CAShapeLayer), layer];
        let fill_rule = NSString::alloc(nil).init_str("even-odd");
        let _: () = msg_send![mask_layer, setFillRule: fill_rule];
        let _: () = msg_send![layer, setMask: mask_layer];

        Some(SwissMask {
            content_view,
            mask_layer,
        })
    }
}

#[cfg(target_os = "macos")]
fn update_window_mask(mask: &SwissMask, size: (usize, usize), holes: &[Hole]) {
    unsafe {
        let bounds: cocoa::foundation::NSRect = msg_send![mask.content_view, bounds];
        let fallback_w = size.0.max(1) as f64;
        let fallback_h = size.1.max(1) as f64;
        let width = bounds.size.width.max(fallback_w);
        let height = bounds.size.height.max(fallback_h);
        let mut path = CGMutablePath::new();
        path.add_rect(
            &CGAffineTransform::identity(),
            CGRect::new(&CGPoint::new(0.0, 0.0), &CGSize::new(width, height)),
        );

        let scale = width.min(height);
        for hole in holes.iter().take(MAX_HOLES) {
            let r = (hole.radius.max(0.01) as f64) * scale;
            let cx = (hole.center[0].clamp(0.0, 1.0) as f64) * width;
            let cy = (hole.center[1].clamp(0.0, 1.0) as f64) * height;
            let rect = CGRect::new(
                &CGPoint::new(cx - r, cy - r),
                &CGSize::new(r * 2.0, r * 2.0),
            );
            path.add_ellipse_in_rect(&CGAffineTransform::identity(), rect);
        }

        let cg_path = path.into_path();
        let _: () = msg_send![mask.mask_layer, setFrame: bounds];
        let _: () = msg_send![mask.mask_layer, setPath: cg_path.as_concrete_TypeRef()];
        let fill_rule = NSString::alloc(nil).init_str("even-odd");
        let _: () = msg_send![mask.mask_layer, setFillRule: fill_rule];
    }
}

pub fn run() {
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
    println!("  N - Neural Overlay Toggle");
    println!("  [ / ] - Neural Gain down/up");
    println!("  Space - Retune neural field weights");
    println!("  C - Collapse swiss-cheese portals");
    println!("  Mouse Left - Spawn a live portal");
    println!("  Mouse Right - Squeeze the particle rim");
    println!("  UP/DOWN - Wind Strength");
    println!("  R - Reset");
    println!("  Esc - Quit");
    println!();

    let mut window = Window::new(
        "Two Sheets - Artistic Camera Effects",
        W,
        H,
        WindowOptions {
            borderless: true,
            title: false,
            resize: true,
            ..WindowOptions::default()
        },
    )
    .unwrap();
    window.limit_update_rate(Some(std::time::Duration::from_micros(16_600)));

    let mut swiss_mask = setup_window_mask(&window);
    let mut holes = seed_default_holes();
    let mut collapse_wave = 1.0f32;
    let mut last_flux = 0.0f32;
    let mut mouse_spawn_armed = false;

    // Audio setup
    let audio_ring: Arc<Mutex<AudioRingBuffer>>;
    let mut audio_enabled: bool;
    let _audio_stream: AudioStream;

    #[cfg(feature = "audio")]
    {
        audio_ring = Arc::new(Mutex::new(AudioRingBuffer::new(8192)));
        match setup_audio_capture(audio_ring.clone()) {
            Ok(stream) => {
                _audio_stream = Some(stream);
                audio_enabled = true;
            }
            Err(e) => {
                eprintln!("⚠️  Audio failed: {}. Continuing without audio.", e);
                _audio_stream = None;
                audio_enabled = false;
            }
        }
    }

    #[cfg(not(feature = "audio"))]
    {
        println!(
            "⚠️  Built without audio capture; run with `--features audio` on macOS to hear the neural feedback."
        );
        audio_ring = Arc::new(Mutex::new(AudioRingBuffer::new(1)));
        _audio_stream = ();
        audio_enabled = false;
    }

    let mut audio_samples = vec![0.0f32; 2048];
    let mut spectrum = [0.0f32; 3];
    let mut last_ai_log = std::time::Instant::now();

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
    let out_buf = device.new_buffer((W * H * 4) as u64, MTLResourceOptions::StorageModeShared);
    let params_buf = device.new_buffer(
        std::mem::size_of::<Params>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let analysis_buf = device.new_buffer(
        std::mem::size_of::<Analysis>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let weights_buf = device.new_buffer(
        std::mem::size_of::<NeuralWeights>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let weights_ptr = weights_buf.contents() as *mut NeuralWeights;
    unsafe {
        weights_ptr.write(init_neural_weights());
        std::ptr::write_bytes(analysis_buf.contents(), 0, std::mem::size_of::<Analysis>());
    }
    let mut zero_copy_logged = false;
    let mut camera_layout_warning = false;
    let mut camera_channels = 3usize;

    let mut time = 0.0f32;
    let mut wind_strength = 1.0f32;
    let mut warp_pattern = 0.0f32;
    let mut show_grid = 0u32;
    let mut camera_mode = 0u32;
    let mut neural_enabled = 1u32;
    let mut neural_gain = 0.65f32;
    let mut weights_anim_logged = false;

    println!("Starting...\n");
    println!("🌊 Pattern: Sine Wave Ripples");
    println!("📷 Camera Mode 0: Centered & Flipped");
    println!("🤖 Neural overlay active (gain {:.2})", neural_gain);

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

        if window.is_key_pressed(Key::N, minifb::KeyRepeat::No) {
            neural_enabled = if neural_enabled == 0 { 1 } else { 0 };
            println!(
                "Neural overlay: {}",
                if neural_enabled == 1 {
                    "ON (shared GPU weights live)"
                } else {
                    "OFF"
                }
            );
        }

        if window.is_key_pressed(Key::LeftBracket, minifb::KeyRepeat::No) {
            neural_gain = (neural_gain - 0.05).max(0.0);
            println!("Neural gain: {:.2}", neural_gain);
        }

        if window.is_key_pressed(Key::RightBracket, minifb::KeyRepeat::No) {
            neural_gain = (neural_gain + 0.05).min(2.0);
            println!("Neural gain: {:.2}", neural_gain);
        }

        let mut mouse_pos = window
            .get_mouse_pos(MouseMode::Clamp)
            .unwrap_or((W as f32 * 0.5, H as f32 * 0.5));
        mouse_pos.0 = (mouse_pos.0 / W as f32).clamp(0.0, 1.0);
        mouse_pos.1 = (mouse_pos.1 / H as f32).clamp(0.0, 1.0);

        collapse_wave += (1.0 - collapse_wave) * 0.04;
        if window.is_key_pressed(Key::C, minifb::KeyRepeat::No) {
            collapse_wave = 0.25;
            println!("🧀 Swiss-cheese collapse pulse triggered");
        }
        if window.get_mouse_down(MouseButton::Right) {
            collapse_wave = (collapse_wave * 0.8).min(0.55);
        }

        if window.is_key_pressed(Key::B, minifb::KeyRepeat::No) {
            let audio_energy = (spectrum[0] + spectrum[1] + spectrum[2]) / 3.0;
            let sketches = spiral_portal_burst(time, audio_energy, last_flux);
            let mut spawned = 0;
            for sketch in sketches {
                holes.push(Hole::new_free(
                    sketch.center[0],
                    sketch.center[1],
                    sketch.radius,
                ));
                spawned += 1;
            }
            if spawned > 0 {
                println!(
                    "🎆 Spawned {} zero-copy firework portals from the SwissMask bloom cache",
                    spawned
                );
            }
        }

        let left_down = window.get_mouse_down(MouseButton::Left);
        if left_down && !mouse_spawn_armed {
            let spawn_radius = 0.07 + (spectrum[0] + spectrum[2]) * 0.06 + last_flux * 0.05;
            holes.push(Hole::new_free(mouse_pos.0, mouse_pos.1, spawn_radius));
            println!(
                "🟡 Spawned live portal at ({:.2}, {:.2}) straight from shared memory",
                mouse_pos.0, mouse_pos.1
            );
            mouse_spawn_armed = true;
        }
        if !left_down {
            mouse_spawn_armed = false;
        }

        let weights_ref = unsafe { &mut *weights_ptr };

        if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
            retune_weights(weights_ref, time);
            println!("🤖 Neural weights retuned directly from CPU into GPU shared memory");
        }

        // Camera capture with CENTER CROP
        let mut camera_frame: Option<(nokhwa::buffer::Buffer, MTLBuffer)> = None;
        if let Ok(frame) = cam.frame() {
            let bytes = frame.buffer();
            let len = bytes.len();
            let pixel_count = cam_w * cam_h;
            if pixel_count == 0 {
                if !camera_layout_warning {
                    eprintln!("⚠️  Camera returned empty frame; skipping");
                    camera_layout_warning = true;
                }
            } else if len % pixel_count != 0 {
                if !camera_layout_warning {
                    eprintln!(
                        "⚠️  Camera buffer size {} is not a multiple of {} pixels; zero-copy path skipped",
                        len, pixel_count
                    );
                    camera_layout_warning = true;
                }
            } else {
                let mut channels = len / pixel_count;
                if channels < 3 {
                    if !camera_layout_warning {
                        eprintln!(
                            "⚠️  Camera reported only {} bytes per pixel; need RGB or RGBA",
                            channels
                        );
                        camera_layout_warning = true;
                    }
                } else {
                    if channels > 4 {
                        channels = 4;
                    }

                    camera_channels = channels;

                    let metal_buffer = device.new_buffer_with_bytes_no_copy(
                        bytes.as_ptr() as *const std::ffi::c_void,
                        len as u64,
                        MTLResourceOptions::StorageModeShared,
                        None,
                    );

                    if !zero_copy_logged {
                        println!(
                            "✅ Zero-copy camera feed established: {}×{} frame, {} channels, {} bytes reused directly as Metal buffer",
                            cam_w,
                            cam_h,
                            camera_channels,
                            len
                        );
                        zero_copy_logged = true;
                    }

                    camera_frame = Some((frame, metal_buffer));
                }
            }
        }

        if camera_frame.is_none() {
            continue;
        }

        let (frame, cam_buf) = camera_frame.unwrap();
        let crop_x = ((cam_w.saturating_sub(W)) / 2) as u32;
        let crop_y = ((cam_h.saturating_sub(H)) / 2) as u32;

        // Audio analysis
        if audio_enabled {
            if let Ok(ring) = audio_ring.lock() {
                ring.get_samples(&mut audio_samples);
                analyze_spectrum(&audio_samples, &mut spectrum);
            }
        } else {
            spectrum = [0.0; 3];
        }

        animate_weights(weights_ref, &spectrum, mouse_pos, time);
        if !weights_anim_logged {
            println!(
                "🔁 Animating neural weights in-place inside shared GPU memory while the GPU reads them each frame"
            );
            weights_anim_logged = true;
        }
        drop(weights_ref);

        let hole_params_array =
            update_hole_field(&mut holes, 0.016, mouse_pos, last_flux, collapse_wave);
        if let Some(mask) = swiss_mask.as_ref() {
            update_window_mask(mask, window.get_size(), &holes);
        }
        let hole_count = holes.len().min(MAX_HOLES) as u32;

        // Update parameters
        let params = Params {
            time,
            w: W as u32,
            h: H as u32,
            cam_w: cam_w as u32,
            cam_h: cam_h as u32,
            cam_stride: cam_w as u32,
            cam_offset_x: crop_x,
            cam_offset_y: crop_y,
            cam_channels: camera_channels as u32,
            wind_strength,
            warp_pattern,
            audio_bass: spectrum[0],
            audio_mid: spectrum[1],
            audio_treble: spectrum[2],
            show_grid,
            camera_mode,
            mouse_x: mouse_pos.0,
            mouse_y: mouse_pos.1,
            neural_gain,
            neural_enabled,
            hole_count,
            hole_pad: [0; 3],
            holes: hole_params_array,
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
            unsafe {
                std::ptr::write_bytes(analysis_buf.contents(), 0, std::mem::size_of::<Analysis>());
            }
            let cmd = q.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();

            enc.set_compute_pipeline_state(&p_render);
            enc.set_buffer(0, Some(&cam_buf), 0);
            enc.set_buffer(1, Some(&out_buf), 0);
            enc.set_buffer(2, Some(&params_buf), 0);
            enc.set_buffer(3, Some(&weights_buf), 0);
            enc.set_buffer(4, Some(&analysis_buf), 0);

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

        let metrics = unsafe { *(analysis_buf.contents() as *const Analysis) };
        last_flux = (metrics.edge_flux as f32) / 65535.0;

        if last_ai_log.elapsed() > std::time::Duration::from_millis(500) {
            let ops = metrics.heavy_ops as f32 / 1_000_000.0;
            let energy = metrics.mlp_energy as f32 / 8192.0;
            let packed = metrics.brightest_activation;
            let intensity = ((packed >> 22) & 0x3ff) as f32 / 1023.0;
            let brightest_x = ((packed >> 11) & 0x7ff) as u32;
            let brightest_y = (packed & 0x7ff) as u32;
            println!(
                "🔥 GPU MLP churn: {:.2}M fused ops, energy {:.2}, brightest activation {} at ({}, {}), cheese flux {:.2}",
                ops,
                energy,
                intensity,
                brightest_x,
                brightest_y,
                last_flux * 10.0
            );
            last_ai_log = std::time::Instant::now();
        }

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

        drop(frame);
    }
}
