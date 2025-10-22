#include <metal_stdlib>
using namespace metal;

inline uint rgb8(float3 c) {
    float3 s = clamp(c, 0.0f, 1.0f) * 255.0f;
    return ((uint)s.x << 16) | ((uint)s.y << 8) | (uint)s.z;
}

inline float3 unpack_rgb(uint v) {
    return float3(float((v >> 16) & 255), float((v >> 8) & 255), float(v & 255)) / 255.0f;
}

struct Params {
    float time;
    uint w;
    uint h;
    float wind_strength;
    float warp_pattern;     // 0-4 for different artistic patterns
    float audio_bass;
    float audio_mid;
    float audio_treble;
    uint show_grid;
    uint camera_mode;       // 0-2 for different camera centering approaches
};

struct SheetProperties {
    float scale;          // Size multiplier
    float fall_speed;     // How fast it falls
    float rotation_speed; // Tumble rate
    float time_offset;    // Phase offset
    float x_offset;       // Horizontal position offset
    float curl_rate;      // How fast it curls/unfurls
    float flap_intensity; // How much it flaps
};

// 3D rotation matrices
inline float3x3 rotateX(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float3x3(
        float3(1.0, 0.0, 0.0),
        float3(0.0, c, -s),
        float3(0.0, s, c)
    );
}

inline float3x3 rotateY(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float3x3(
        float3(c, 0.0, s),
        float3(0.0, 1.0, 0.0),
        float3(-s, 0.0, c)
    );
}

inline float3x3 rotateZ(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float3x3(
        float3(c, -s, 0.0),
        float3(s, c, 0.0),
        float3(0.0, 0.0, 1.0)
    );
}

// Simple 2D Perlin-style noise (hash-based)
inline float hash(float2 p) {
    float h = dot(p, float2(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

inline float noise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);  // Smoothstep

    float a = hash(i);
    float b = hash(i + float2(1.0, 0.0));
    float c = hash(i + float2(0.0, 1.0));
    float d = hash(i + float2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Fractal Brownian Motion - multiple octaves of noise
inline float fbm(float2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// Compute curl of a 2D potential field (creates divergence-free flow)
inline float2 curl_noise(float2 p, float time) {
    float eps = 0.01;

    // Sample noise at offset positions to compute derivatives
    float n1 = fbm(p + float2(0.0, eps), 3);
    float n2 = fbm(p - float2(0.0, eps), 3);
    float n3 = fbm(p + float2(eps, 0.0), 3);
    float n4 = fbm(p - float2(eps, 0.0), 3);

    // Curl = (∂/∂y, -∂/∂x)
    float dx = (n1 - n2) / (2.0 * eps);
    float dy = (n3 - n4) / (2.0 * eps);

    return float2(dx, -dy);
}

// Apply artistic warp pattern
inline float2 apply_warp(float2 pos, Params P) {
    int pattern = int(P.warp_pattern);
    float2 warped = pos;

    if (pattern == 0) {
        // Sine Wave Ripples
        float wave = sin(pos.x * 6.0 + P.time * 2.0 + P.audio_bass * 3.0) * P.wind_strength * 0.15;
        wave += sin(pos.y * 4.0 - P.time * 1.5 + P.audio_mid * 2.0) * P.wind_strength * 0.1;
        warped += float2(wave * 0.5, wave);

    } else if (pattern == 1) {
        // Spiral Vortex
        float dist = length(pos);
        float angle = atan2(pos.y, pos.x);
        float spiral = angle + dist * 3.0 - P.time + P.audio_treble * 2.0;
        float twist = sin(spiral) * P.wind_strength * 0.2;
        warped = pos + float2(cos(angle), sin(angle)) * twist;

    } else if (pattern == 2) {
        // Spherical Bulge
        float dist = length(pos);
        float bulge = (1.0 - dist) * P.wind_strength * 0.3;
        bulge *= (1.0 + P.audio_bass * 0.5);
        warped = pos * (1.0 + bulge);

    } else if (pattern == 3) {
        // Checkerboard Twist
        float checker = sin(pos.x * 10.0 + P.time) * sin(pos.y * 10.0 - P.time);
        checker *= P.wind_strength * 0.15 * (1.0 + P.audio_mid * 0.3);
        warped += float2(checker, -checker * 0.5);

    } else {
        // Hyperbolic Flow
        float flow_x = pos.x / (1.0 + abs(pos.y) + P.audio_treble * 0.3);
        float flow_y = pos.y / (1.0 + abs(pos.x) + P.audio_bass * 0.3);
        warped = float2(flow_x, flow_y) * (1.0 + P.wind_strength * 0.3);
    }

    return warped;
}

// Render a single 2D scrolling, flapping, translucent sheet
inline float4 render_sheet(float2 centered,
                           constant Params& P,
                           device const uint* camera,
                           SheetProperties props,
                           int sheet_id) {

    // Simple 2D positioning - falling animation
    float fall_cycle = fract(P.time * props.fall_speed + props.time_offset);
    float fall_y = 1.5 - fall_cycle * 3.0;  // Falls from top to bottom

    // 2D position (closer to camera, spread horizontally)
    float2 sheet_center = float2(props.x_offset, fall_y);

    // ROTATION - sheets spin as they fall!
    float rotation_angle = P.time * props.rotation_speed + props.time_offset;
    float cos_rot = cos(rotation_angle);
    float sin_rot = sin(rotation_angle);

    // Rotate the centered coordinate around the sheet center
    float2 to_center = centered - sheet_center;
    float2 rotated_pos;
    rotated_pos.x = to_center.x * cos_rot - to_center.y * sin_rot;
    rotated_pos.y = to_center.x * sin_rot + to_center.y * cos_rot;
    float2 rotated_centered = sheet_center + rotated_pos;

    // Sheet bounds (simple 2D rectangle)
    float half_w = 0.4 * props.scale;
    float half_h = 0.5 * props.scale;

    // TURBULENCE - sheets move through invisible fluid currents (using rotated coords)
    float2 turbulence_pos = rotated_centered * 3.0 + P.time * 0.3;
    float2 turbulence = curl_noise(turbulence_pos, P.time) * 0.15;
    turbulence *= P.wind_strength;  // Wind strength controls turbulence intensity

    // VORTICITY FIELD - DISABLED (was creating visible vortex spot)
    // float2 vortex_center = float2(sin(P.time * 0.5) * 0.8, cos(P.time * 0.4) * 0.6);
    // float2 to_vortex = centered - vortex_center;
    // float vortex_dist = length(to_vortex);
    // float vortex_strength = exp(-vortex_dist * vortex_dist) * 0.2;
    // float2 vortex_flow = float2(-to_vortex.y, to_vortex.x) * vortex_strength;
    float2 vortex_flow = float2(0.0, 0.0);  // Disabled

    // FLUID DRAG - DISABLED for cleaner motion
    // float2 velocity = float2(sin(P.time * 2.0 + props.time_offset), cos(P.time * 1.5 + props.time_offset));
    // float drag_coefficient = 0.1;
    // float2 drag = -velocity * drag_coefficient;
    float2 drag = float2(0.0, 0.0);  // Disabled

    // BEDSHEET FLAPPING - add wave displacement to the sheet position (using rotated coords)
    // Multiple wave frequencies create realistic cloth motion
    float wave1 = sin(rotated_centered.x * 5.0 + P.time * 3.0 + props.time_offset) * props.flap_intensity * 0.08;
    float wave2 = sin(rotated_centered.y * 7.0 - P.time * 4.0 + props.time_offset) * props.flap_intensity * 0.06;
    float wave3 = sin((rotated_centered.x + rotated_centered.y) * 4.0 + P.time * 5.0 + props.time_offset) * props.flap_intensity * 0.05;

    // Audio makes the flapping more intense
    float audio_boost = 1.0 + P.audio_treble * 0.5 + P.audio_bass * 0.3;

    // POWER LAW DISTORTIONS - distance-based warping with non-linear falloff (using rotated coords)
    float dist_from_center = length(rotated_centered - sheet_center);

    // CENTER STILLNESS FACTOR - center of sheet stays calm for better reflection
    // Use smooth falloff: center = 0 (no motion), edges = 1 (full motion)
    float center_stillness = smoothstep(0.0, 0.4, dist_from_center);  // Center 40% of sheet is very still

    // Inverse square attraction (like gravity) - subtle pull toward center (using rotated coords)
    float gravity_pull = 1.0 / (1.0 + dist_from_center * dist_from_center * 2.0);
    float2 to_center_rot = normalize(sheet_center - rotated_centered);
    float2 gravity_offset = to_center_rot * gravity_pull * 0.03 * sin(P.time * 0.5 + props.time_offset);

    // Power-3 edge repulsion (corners push outward) (using rotated coords)
    float edge_push = pow(dist_from_center, 3.0) * 0.002;
    float2 from_center_rot = normalize(rotated_centered - sheet_center);
    float2 push_offset = from_center_rot * edge_push * (1.0 + P.audio_mid * 0.3);

    // COMBINE ALL PHYSICS FORCES - apply center_stillness to reduce motion in center
    float2 wave_offset = float2(wave1 + wave3, wave2 + wave3) * audio_boost * center_stillness;
    float2 physics_offset = (turbulence + vortex_flow + drag) * center_stillness;

    // Total displacement combines waves, power laws, and fluid dynamics
    float2 displaced_center = sheet_center + wave_offset + gravity_offset + push_offset + physics_offset;

    // Check if pixel is within sheet bounds (using displaced position)
    float2 local_pos = centered - displaced_center;

    // GENEROUS edge fade instead of hard cutoff
    float fade_margin = 0.15;  // Fade zone width
    float x_fade = 1.0;
    float y_fade = 1.0;

    // X edge fade
    float abs_x = abs(local_pos.x);
    if (abs_x > half_w + fade_margin) {
        return float4(0, 0, 0, 0);  // Completely outside
    } else if (abs_x > half_w) {
        x_fade = 1.0 - smoothstep(half_w, half_w + fade_margin, abs_x);
    }

    // Y edge fade
    float abs_y = abs(local_pos.y);
    if (abs_y > half_h + fade_margin) {
        return float4(0, 0, 0, 0);  // Completely outside
    } else if (abs_y > half_h) {
        y_fade = 1.0 - smoothstep(half_h, half_h + fade_margin, abs_y);
    }

    float edge_fade = x_fade * y_fade;

    // UV coordinates (-1 to 1)
    float2 sheet_uv = float2(local_pos.x / half_w, local_pos.y / half_h);

    // ULTRA MINIMAL DISTORTION - barely any, for clearest reflection
    float2 abs_uv = abs(sheet_uv);
    float2 to_corner = abs_uv - 0.95;  // Only affect extreme corners
    float corner_dist = max(to_corner.x, to_corner.y);

    if (corner_dist > -0.05) {
        float corner_proximity = 1.0 - smoothstep(-0.05, 0.0, corner_dist);

        // Tiny corner wobble
        float wobble = sin(P.time * 3.0 + props.time_offset) * corner_proximity * 0.005;  // Was 0.02
        sheet_uv += sign(sheet_uv) * wobble;
    }

    // GENTLE SCROLL CURL - center has stillness, edges have gentle motion
    float curl_cycle = sin(P.time * props.curl_rate + props.time_offset * 2.0) * 0.5 + 0.5;
    float curl_amount = curl_cycle * 0.15;  // Reduced from 0.3
    float edge_x = abs(sheet_uv.x);

    // CENTER STILLNESS - only edges curl, center stays perfectly still
    // Use power function to keep center flat and only affect outer regions
    float edge_falloff = edge_x * edge_x * edge_x;  // Cubic falloff: center stays at 0, edges move
    float curl = edge_falloff * curl_amount;

    float2 curled_uv = sheet_uv;
    curled_uv.x *= (1.0 - curl * 0.1);  // Less compression (was 0.2)
    curled_uv.y += sin(edge_x * 3.14159 * curl) * 0.02;  // Less lifting (was 0.05)

    // Edge factor for other effects
    float edge_factor = length(sheet_uv);

    // NO WARPING - see yourself clearly
    float2 warped = curled_uv;

    // CAMERA CENTERING MODES - different approaches (keys 1-3)
    float2 cam_uv;

    if (P.camera_mode == 0) {
        // Mode 0: Centered crop with aspect-aware scaling
        // Assumes camera is 1920x1080, output is 1280x720
        float cam_aspect = 1920.0 / 1080.0;  // 16:9
        float out_aspect = float(P.w) / float(P.h);  // Also 16:9

        // Center the warped coordinates
        float2 centered_uv = warped * 0.5 + 0.5;  // Standard mapping

        // Apply horizontal flip for mirror effect
        centered_uv.x = 1.0 - centered_uv.x;

        cam_uv = centered_uv;

    } else if (P.camera_mode == 1) {
        // Mode 1: Zoom out more, NOW FLIPPED
        float2 zoom_uv = warped * 0.35 + 0.5;
        zoom_uv.x = 1.0 - zoom_uv.x;  // Mirror
        cam_uv = zoom_uv;

    } else {
        // Mode 2: Tight crop, centered, flipped
        float2 tight_uv = warped * 0.6 + 0.5;
        tight_uv.x = 1.0 - tight_uv.x;  // Mirror

        // Offset to center face (adjust Y down slightly)
        tight_uv.y += 0.05;

        cam_uv = tight_uv;
    }

    if (cam_uv.x < 0.0 || cam_uv.x > 1.0 || cam_uv.y < 0.0 || cam_uv.y > 1.0) {
        return float4(0, 0, 0, 0);
    }

    int2 cam_coord = int2(cam_uv * float2(P.w - 1, P.h - 1));
    cam_coord = clamp(cam_coord, int2(0), int2(P.w - 1, P.h - 1));
    float3 color = unpack_rgb(camera[cam_coord.y * int(P.w) + cam_coord.x]);

    // Blotches removed from here - now rendered in main kernel to avoid duplication

    // TRANSLUCENCY
    float base_alpha = 0.75;

    // Edges more transparent
    float edge_alpha = 1.0 - edge_factor * 0.3;

    // Curled parts more opaque
    float curl_alpha = 1.0 + curl * 0.2;

    // Falling animation affects transparency (higher sheets more transparent)
    float fall_alpha = 0.7 + fall_cycle * 0.3;

    // GENEROUS vertical screen-edge fade - fade in/out as sheets enter/exit screen
    float vertical_fade = 1.0;
    if (fall_y > 1.0) {
        // Entering from top
        vertical_fade = 1.0 - smoothstep(1.0, 1.5, fall_y);
    } else if (fall_y < -1.0) {
        // Exiting below
        vertical_fade = smoothstep(-1.5, -1.0, fall_y);
    }

    float alpha = base_alpha * edge_alpha * curl_alpha * fall_alpha * edge_fade * vertical_fade;

    // Grid overlay
    if (P.show_grid > 0) {
        float2 grid_uv = warped * 15.0;
        if (fract(grid_uv.x) < 0.05 || fract(grid_uv.y) < 0.05) {
            float grid = 0.3 + P.audio_mid * 0.4;
            color = mix(color, float3(0.5, 0.8, 1.0), grid);
        }
    }

    return float4(color, alpha);
}

// Two sheets with artistic wind effects
kernel void render_two_sheets(device const uint* camera     [[buffer(0)]],
                              device uint* output            [[buffer(1)]],
                              constant Params& P             [[buffer(2)]],
                              uint2 gid                      [[thread_position_in_grid]])
{
    if (gid.x >= P.w || gid.y >= P.h) return;

    // Screen coordinates
    float2 uv = float2(gid) / float2(P.w, P.h);
    float2 centered = uv * 2.0 - 1.0;
    centered.x *= float(P.w) / float(P.h);

    // Clean background - no warp, just straight camera (FLIPPED)
    float2 bg_uv = uv;
    bg_uv.x = 1.0 - bg_uv.x;  // Flip horizontally for mirror effect

    // Sample camera for background
    float2 bg_cam_uv = bg_uv;
    int2 bg_cam_coord = int2(bg_cam_uv * float2(P.w - 1, P.h - 1));
    bg_cam_coord = clamp(bg_cam_coord, int2(0), int2(P.w - 1, P.h - 1));
    float3 bg_texture = unpack_rgb(camera[bg_cam_coord.y * int(P.w) + bg_cam_coord.x]);

    // Darken and tint background
    float3 color = bg_texture * 0.15;
    color += float3(0.05, 0.05, 0.1);
    color += float3(0.1, 0.05, 0.2) * P.audio_bass * 0.2;

    // KELP FOREST - delicate strands growing upward from bottom in organic clusters!
    const int num_strings = 25;
    float total_audio = P.audio_bass + P.audio_mid + P.audio_treble;

    float2 string_uv = uv;

    for (int string_id = 0; string_id < num_strings; string_id++) {
        float string_offset = float(string_id) / float(num_strings);

        // Each string has its own personality
        float string_chaos = sin(float(string_id) * 2.7) * 0.5 + 0.5;
        float string_speed = 0.8 + string_chaos * 1.2;

        // SIDE-ANCHORED KELP - Left side = BASS responsive, Right side = TREBLE responsive
        // Determine which side this strand is on
        bool is_left_side = (string_id < num_strings / 2);

        float anchor_y;
        float anchor_x_edge;
        float string_length = 0.2 + string_chaos * 0.17;  // Horizontal length reaching inward

        if (is_left_side) {
            // LEFT SIDE - anchored on left edge, grows rightward (responds to BASS)
            anchor_x_edge = 0.0;  // All the way to left edge!
            // Spread vertically along left edge
            anchor_y = 0.2 + (float(string_id) / float(num_strings / 2)) * 0.6;
        } else {
            // RIGHT SIDE - anchored on right edge, grows leftward (responds to TREBLE)
            anchor_x_edge = 1.0;  // All the way to right edge!
            // Spread vertically along right edge
            anchor_y = 0.2 + (float(string_id - num_strings / 2) / float(num_strings / 2)) * 0.6;
        }

        float2 anchor = float2(anchor_x_edge, anchor_y);

        // Calculate position along string (t) - parametric variable from anchor to tip
        // For left side: grows right, for right side: grows left
        float t;
        if (is_left_side) {
            t = (string_uv.x - anchor.x) / string_length;  // Left to right
        } else {
            t = (anchor.x - string_uv.x) / string_length;  // Right to left
        }

        // Skip pixels outside the kelp strand entirely - this prevents straight lines!
        if (t < 0.0 || t > 1.0) continue;  // Not on this kelp strand, try next one

        // SIDE-SPECIFIC AUDIO RESPONSE - Left = BASS, Right = TREBLE
        float primary_audio = is_left_side ? P.audio_bass : P.audio_treble;
        float string_influence = sin(float(string_id) * 1.5) * 0.5 + 0.5;

        // HORIZONTAL KELP PHYSICS - growing inward from sides with vertical sway

        // Base X position grows horizontally from anchor
        float base_x;
        if (is_left_side) {
            base_x = anchor.x + t * string_length;  // Grow rightward
        } else {
            base_x = anchor.x - t * string_length;  // Grow leftward
        }

        // AMBIENT FLOW - gentle vertical drift like underwater current
        float ambient_drift_y = sin(P.time * 0.5 + string_offset * 6.28) * 0.03;

        // VERTICAL SWAY - up/down motion that increases toward tip
        // The further from anchor (higher t), the more it sways
        float sway_amount = t * t;  // Quadratic - tip moves much more than root

        // Multiple frequencies of vertical sway
        float sway1 = sin(P.time * string_speed * 0.7 - t * 3.0 + string_offset * 6.28) * sway_amount * 0.08;
        float sway2 = sin(P.time * string_speed * 1.4 - t * 5.0 + string_offset * 3.14) * sway_amount * 0.05;
        float sway3 = cos(P.time * string_speed * 2.1 + t * 4.0 + string_offset * 1.57) * sway_amount * 0.03;

        float base_sway_y = sway1 + sway2 + sway3 + ambient_drift_y;

        // AUDIO-REACTIVE VERTICAL DISPLACEMENT - perpendicular waves along the strand
        // Left side: BASS creates big slow waves
        // Right side: TREBLE creates fast shimmering
        float audio_wave;
        float audio_push_y;
        float audio_push_x = 0.0;  // Horizontal displacement from audio

        if (is_left_side) {
            // BASS - slow, POWERFUL rolling waves with depth
            // Multiple bass wave components for rich movement
            float bass_wave1 = sin(P.time * 1.8 - t * 6.0 + string_offset * 6.28);
            float bass_wave2 = sin(P.time * 2.5 - t * 10.0 + string_offset * 4.71);
            float bass_swell = (bass_wave1 + bass_wave2 * 0.6);

            // Bass creates big vertical swells
            float bass_strength = P.audio_bass * P.audio_bass;  // Square for more dramatic response
            audio_push_y = bass_swell * bass_strength * string_influence * 0.35 * sway_amount;

            // Bass also pushes horizontally inward (like a pressure wave)
            audio_push_x = bass_wave1 * bass_strength * string_influence * 0.08 * sway_amount;
        } else {
            // TREBLE - fast, SHARP shimmering with aggressive tip whipping
            // High frequency vibrations
            float treble_shimmer = sin(P.time * 12.0 - t * 25.0 + string_offset * 1.57);
            float treble_ripple = sin(P.time * 18.0 - t * 35.0 + string_offset * 2.35);

            float tip_whip_factor = t * t * t * t;  // Quartic - VERY tip-focused
            float treble_intensity = P.audio_treble * (1.0 + P.audio_treble * 0.5);  // Amplify high treble

            // Fast vibrating shimmer
            float treble_base_motion = (treble_shimmer + treble_ripple * 0.4) * treble_intensity * string_influence * 0.12 * sway_amount;

            // Aggressive tip whipping for high frequencies
            float treble_whip = treble_shimmer * treble_intensity * string_influence * 0.45 * tip_whip_factor;

            audio_push_y = treble_base_motion + treble_whip;

            // Treble creates rapid horizontal flutter
            audio_push_x = treble_ripple * treble_intensity * string_influence * 0.06 * t * t;
        }

        // Mid frequencies affect both sides - medium-speed pulses
        float mid_wave = sin(P.time * 5.0 - t * 14.0 + string_offset * 3.14);
        float mid_push_y = mid_wave * P.audio_mid * string_influence * 0.12 * sway_amount;

        // LIVELY TURBULENCE - strands get buffeted by currents
        float2 turb_pos = float2(t * 5.0, P.time * 1.2 + string_offset * 3.0);
        float turb_y = (noise(turb_pos) - 0.5) * 0.06 * sway_amount;
        float turb_x = (noise(turb_pos * 1.3 + float2(2.7, 1.1)) - 0.5) * 0.04 * sway_amount;

        // TOTAL POSITION - combine all motions including audio-reactive horizontal displacement
        float string_x = base_x + audio_push_x + turb_x;
        float string_y = anchor.y + base_sway_y + audio_push_y + mid_push_y + turb_y;

        // PERPENDICULAR DISTANCE - only measure vertical distance to keep strings sharp!
        // This prevents motion blur/smearing from vertical movement
        float dist_to_string = abs(string_uv.y - string_y);

        // Varying thickness - 50% thinner for delicate kelp
        float base_thickness = 0.0005 + string_chaos * 0.00025;

        if (dist_to_string < base_thickness * 15.0) {  // Tighter for delicate kelp
            float core_dist = dist_to_string / base_thickness;

            // DELICATE KELP RENDERING - tight falloff for fine strands
            float core = smoothstep(2.0, 0.1, core_dist);
            float inner_glow = smoothstep(6.0, 0.6, core_dist) * 0.6;
            float outer_glow = smoothstep(15.0, 3.0, core_dist) * 0.3;

            float base_intensity = core + inner_glow + outer_glow;

            // AUDIO-REACTIVE INTENSITY PULSES - side-specific audio response
            float audio_pulse = sin(P.time * 4.0 - t * 8.0) * primary_audio * string_influence;

            float audio_intensity_mod = 1.0 + audio_pulse * 0.5;
            float intensity = base_intensity * audio_intensity_mod;

            // VIBRANT FIRE GRADIENT - more saturated and lively
            float3 color_anchor = float3(0.9, 0.15, 0.05);     // Bright red at anchor
            float3 color_middle = float3(1.0, 0.6, 0.15);      // Vibrant orange
            float3 color_tip = float3(1.0, 0.95, 0.6);         // Bright yellow-white

            // Smooth gradient with audio-reactive color shift
            float color_t = t + sin(P.time * 2.0 + string_offset * 6.28) * total_audio * 0.1;
            color_t = clamp(color_t, 0.0, 1.0);

            float3 fire_color;
            if (color_t < 0.5) {
                fire_color = mix(color_anchor, color_middle, smoothstep(0.0, 0.5, color_t));
            } else {
                fire_color = mix(color_middle, color_tip, smoothstep(0.5, 1.0, color_t));
            }

            // DRAMATIC AUDIO SPECTRUM COLORING - side-specific frequency emphasis
            if (is_left_side) {
                // Left side (BASS) - deep reds and warm orange pulses
                float bass_power = P.audio_bass * P.audio_bass;  // Square for dramatic effect

                // Bass pushes toward deep red
                fire_color.r = mix(fire_color.r, 1.0, bass_power * string_influence * 0.5);
                fire_color.g = mix(fire_color.g, 0.7, bass_power * string_influence * 0.3);

                // Pulsing warmth from bass
                float bass_glow = bass_power * string_influence * (0.5 + 0.5 * sin(P.time * 2.5 - t * 4.0));
                fire_color.r += bass_glow * 0.3;
                fire_color.g += bass_glow * 0.15;
            } else {
                // Right side (TREBLE) - bright cyan/white shimmers
                float treble_sharp = P.audio_treble * (1.0 + P.audio_treble * 0.7);  // Amplify high treble

                // Treble creates cyan-white flashes
                fire_color.b = mix(fire_color.b, 1.0, treble_sharp * string_influence * 0.6);
                fire_color.g = mix(fire_color.g, 1.0, treble_sharp * string_influence * 0.4);

                // High-frequency sparkle effect on tips
                float tip_sparkle = treble_sharp * string_influence * t * t * (0.5 + 0.5 * sin(P.time * 15.0 - t * 20.0));
                fire_color.b += tip_sparkle * 0.4;
                fire_color.r += tip_sparkle * 0.2;  // Slight magenta tint on sparkles
            }

            // Overall brightness from total audio
            fire_color *= (1.0 + total_audio * 0.8);

            // Minimum ember glow when quiet
            fire_color = mix(float3(0.3, 0.18, 0.1), fire_color, smoothstep(0.0, 0.2, total_audio));

            // ENERGETIC BASE LEVEL - always lively
            float energy = 0.6 + total_audio * 4.0;

            // Add fire color with dynamic intensity
            color += fire_color * intensity * energy * 0.5;

            // Gentle sparkle when sound peaks (less harsh)
            if (total_audio > 0.7) {
                float flash = core * smoothstep(0.7, 1.0, total_audio) * 0.5;
                color += float3(1.0, 0.95, 0.85) * flash;
            }
        }
    }

    // TINY BLOTCH REMOVED - keeping just the falling sheets and underwater strings

    // Define 6 sheets with varying properties - spread horizontally, bigger
    SheetProperties sheets[6];

    // 4 MEDIUM SHEETS - better size, clear reflection, calm motion
    // Sheet 0: Far left
    sheets[0].scale = 0.9;  // Medium-large
    sheets[0].fall_speed = 0.08;  // Slow, calmer
    sheets[0].rotation_speed = 0.3;  // Less rotation for clearer reflection
    sheets[0].time_offset = 0.0;
    sheets[0].x_offset = -1.2;  // Spread out
    sheets[0].curl_rate = 0.12;  // Less curling for better reflection
    sheets[0].flap_intensity = 0.12;  // Gentler flapping

    // Sheet 1: Left-center
    sheets[1].scale = 1.0;  // Slightly larger
    sheets[1].fall_speed = 0.10;
    sheets[1].rotation_speed = 0.35;
    sheets[1].time_offset = 4.0;  // Staggered
    sheets[1].x_offset = -0.5;
    sheets[1].curl_rate = 0.15;
    sheets[1].flap_intensity = 0.15;

    // Sheet 2: Right-center
    sheets[2].scale = 0.95;
    sheets[2].fall_speed = 0.09;
    sheets[2].rotation_speed = 0.32;
    sheets[2].time_offset = 8.0;  // More staggered
    sheets[2].x_offset = 0.5;
    sheets[2].curl_rate = 0.13;
    sheets[2].flap_intensity = 0.13;

    // Sheet 3: Far right
    sheets[3].scale = 1.05;  // Largest
    sheets[3].fall_speed = 0.11;
    sheets[3].rotation_speed = 0.38;
    sheets[3].time_offset = 12.0;
    sheets[3].x_offset = 1.2;
    sheets[3].curl_rate = 0.16;
    sheets[3].flap_intensity = 0.16;

    // Sheet 4: Right
    sheets[4].scale = 1.5;
    sheets[4].fall_speed = 0.16;  // Different speed
    sheets[4].rotation_speed = 1.7;
    sheets[4].time_offset = 12.0;  // Much more staggered (5.0 → 12.0)
    sheets[4].x_offset = 1.3;  // More spread (1.0 → 1.3)
    sheets[4].curl_rate = 0.42;
    sheets[4].flap_intensity = 0.6;

    // Sheet 5: Far right
    sheets[5].scale = 1.4;
    sheets[5].fall_speed = 0.12;  // Slower
    sheets[5].rotation_speed = 1.0;
    sheets[5].time_offset = 15.0;  // Much more staggered (0.8 → 15.0)
    sheets[5].x_offset = 2.0;  // More spread (1.6 → 2.0)
    sheets[5].curl_rate = 0.38;
    sheets[5].flap_intensity = 0.4;

    // Render 4 medium sheets with better reflection (indices 0-3 only)
    for (int i = 3; i >= 0; i--) {
        float4 sheet_color = render_sheet(centered, P, camera, sheets[i], i);

        if (sheet_color.a > 0.01) {
            // Alpha blend: C_out = C_fg * a + C_bg * (1 - a)
            color = color * (1.0 - sheet_color.a) + sheet_color.rgb * sheet_color.a;
        }
    }

    output[gid.y * P.w + gid.x] = rgb8(clamp(color, 0.0f, 1.0f));
}
