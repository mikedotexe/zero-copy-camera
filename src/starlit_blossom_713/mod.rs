use std::f32::consts::TAU;

#[derive(Clone, Copy, Debug)]
pub struct PortalSketch {
    pub center: [f32; 2],
    pub radius: f32,
}

impl PortalSketch {
    pub fn new(x: f32, y: f32, radius: f32) -> Self {
        Self {
            center: [x, y],
            radius,
        }
    }
}

pub fn spiral_portal_burst(time: f32, audio_energy: f32, flux_energy: f32) -> Vec<PortalSketch> {
    let mut sketches = Vec::with_capacity(18);
    let swirl = (time * 0.45 + flux_energy * 0.7).sin();
    let swell = (audio_energy * 3.2).clamp(0.0, 3.2);

    let rings = 3;
    for ring in 0..rings {
        let ring_mix = ring as f32 / rings as f32;
        let ring_radius = 0.18 + ring_mix * 0.22 + swell * 0.03;
        let count = 5 + ring * 2;
        for i in 0..count {
            let progress = i as f32 / count as f32;
            let angle =
                progress * TAU + swirl * (0.6 + ring_mix * 0.8) + time * (0.12 + ring_mix * 0.07);
            let jitter = (time * 0.4 + progress * 13.37).sin() * 0.015;
            let x = 0.5 + angle.cos() * ring_radius + jitter * (0.4 + flux_energy * 0.2);
            let y = 0.5 + angle.sin() * ring_radius + jitter * (0.3 + audio_energy * 0.5);
            let radius =
                (0.035 + ring_mix * 0.03 + swell * 0.01 + flux_energy * 0.02).clamp(0.03, 0.17);
            sketches.push(PortalSketch::new(x, y, radius));
        }
    }

    let portal_line = 6;
    for i in 0..portal_line {
        let t = i as f32 / (portal_line - 1).max(1) as f32;
        let wave = (time * 0.9 + t * 8.0).sin();
        let radius = (0.04 + wave.abs() * 0.02 + flux_energy * 0.015).clamp(0.03, 0.12);
        let x = 0.12 + t * 0.76 + wave * 0.08;
        let y = 0.18 + wave * 0.05 + audio_energy * 0.07;
        sketches.push(PortalSketch::new(x, y, radius));
    }

    sketches
}
