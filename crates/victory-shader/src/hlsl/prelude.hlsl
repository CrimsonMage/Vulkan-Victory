// Helpers for translated GFX10 code (victory-shader `hlsl`): each is one
// console operation as the hardware computes it, where HLSL's own differs.
// Registers are uint; floats travel as their bits.

// v_max_f32 / v_min_f32: -0 orders below +0.
float hw_max(float a, float b) {
  float r = max(a, b);
  uint ia = asuint(a), ib = asuint(b);
  return ((ia | ib) & 0x7fffffffu) == 0u ? asfloat(ia & ib) : r;
}
float hw_min(float a, float b) {
  float r = min(a, b);
  uint ia = asuint(a), ib = asuint(b);
  return ((ia | ib) & 0x7fffffffu) == 0u ? asfloat(ia | ib) : r;
}
// 0 x anything = 0 (D3D9 multiply).
float hw_mul_legacy(float a, float b) {
  return (a == 0.0f || b == 0.0f) ? 0.0f : a * b;
}
bool hw_isnan(float f) { return (asuint(f) & 0x7fffffffu) > 0x7f800000u; }
// The median of three; with a NaN among them the minimum.
float hw_med3(float a, float b, float c) {
  float lo = min(a, b), hi = max(a, b);
  float med = max(lo, min(hi, c));
  return (hw_isnan(a) || hw_isnan(b) || hw_isnan(c)) ? min(lo, c) : med;
}
// a - floor(a), never reaching 1.0.
float hw_fract(float a) {
  float r = frac(a);
  return r >= 1.0f ? asfloat(0x3f7fffffu) : r;
}
// Sine and cosine of an angle in turns.
float hw_sin(float x) {
  float part = x == 0.0f ? x : x - round(x);
  float r = sin(part * 6.2831853f);
  return abs(part) == 0.5f ? 0.0f : r;
}
float hw_cos(float x) {
  float part = x == 0.0f ? x : x - round(x);
  float r = cos(part * 6.2831853f);
  return abs(part) == 0.25f ? 0.0f : r;
}
// Bit-field extract; widths from 32 on take every bit.
uint hw_bfe_u(uint a, uint off, uint w) {
  uint s = a >> (off & 31u);
  uint m = w >= 32u ? 0xffffffffu : ((1u << (w & 31u)) - 1u);
  return s & m;
}
uint hw_bfe_i(uint a, uint off, uint w) {
  uint s = uint(int(a) >> (off & 31u));
  if (w >= 32u) return s;
  uint wl = w & 31u;
  uint m = (1u << wl) - 1u;
  uint r = s & m;
  if (wl == 0u) return r;
  uint left = (32u - wl) & 31u;
  return uint(int(r << left) >> left);
}
uint hw_mul_hi_u(uint a, uint b) { return uint((uint64_t(a) * uint64_t(b)) >> 32); }
uint hw_mul_hi_i(uint a, uint b) { return uint(uint64_t(int64_t(int(a)) * int64_t(int(b))) >> 32); }
uint hw_alignbit(uint a, uint b, uint sh) {
  sh &= 31u;
  return (b >> sh) | (sh == 0u ? 0u : (a << ((32u - sh) & 31u)));
}
uint hw_perm_byte(uint a, uint b, uint sel) {
  if (sel < 8u) {
    uint w = sel >= 4u ? a : b;
    return (w >> ((sel & 3u) * 8u)) & 0xffu;
  }
  if (sel < 12u) {
    uint w = sel < 10u ? b : a;
    uint bit = (sel & 1u) != 0u ? 31u : 15u;
    return uint(int(w << (31u - bit)) >> 31) & 0xffu;
  }
  return sel == 12u ? 0u : 0xffu;
}
uint hw_perm(uint a, uint b, uint sel) {
  return hw_perm_byte(a, b, sel & 0xffu) | (hw_perm_byte(a, b, (sel >> 8) & 0xffu) << 8)
       | (hw_perm_byte(a, b, (sel >> 16) & 0xffu) << 16) | (hw_perm_byte(a, b, sel >> 24) << 24);
}
// f32 -> f16 bits, toward zero (v_cvt_pkrtz).
uint hw_f16_rtz(float f) {
  uint h = f32tof16(f);
  float back = f16tof32(h);
  return abs(back) > abs(f) ? h - 1u : h;
}
// f32 -> f16 bits, nearest even, whatever the host's f32tof16 rounds.
uint hw_f16_rne(float f) {
  uint lo = hw_f16_rtz(f);
  uint hi = lo + 1u;
  float below = abs(f16tof32(lo));
  float above = (hi & 0x7fffu) == 0x7c00u ? 65536.0f : abs(f16tof32(hi));
  float fa = abs(f);
  float under = fa - below, over = above - fa;
  bool up = over < under || (over == under && (lo & 1u) != 0u);
  return up ? hi : lo;
}
uint hw_cvt_u32(float f) { return hw_isnan(f) ? 0u : (f <= 0.0f ? 0u : (f >= 4294967296.0f ? 0xffffffffu : uint(f))); }
uint hw_cvt_i32(float f) { return hw_isnan(f) ? 0u : (f <= -2147483648.0f ? 0x80000000u : (f >= 2147483648.0f ? 0x7fffffffu : uint(int(f)))); }
// Cube-map face coordinates (major axis z, then y, then x).
float hw_cube_pick(float x, float y, float z, float zv, float yv, float xv) {
  float ax = abs(x), ay = abs(y), az = abs(z);
  return (az >= ax && az >= ay) ? zv : (ay >= ax ? yv : xv);
}
float hw_cubeid(float x, float y, float z) {
  return hw_cube_pick(x, y, z, z < 0.0f ? 5.0f : 4.0f, y < 0.0f ? 3.0f : 2.0f, x < 0.0f ? 1.0f : 0.0f);
}
float hw_cubesc(float x, float y, float z) {
  return hw_cube_pick(x, y, z, z < 0.0f ? -x : x, x, x < 0.0f ? z : -z);
}
float hw_cubetc(float x, float y, float z) {
  return hw_cube_pick(x, y, z, -y, y < 0.0f ? -z : z, -y);
}
float hw_cubema(float x, float y, float z) {
  return 2.0f * hw_cube_pick(x, y, z, z, y, x);
}
// The direction a cube sample's face coordinates (s, t in [1, 2], face id) stand for.
float3 hw_cube_dir(float s, float t, float face) {
  float u = s * 2.0f - 3.0f, v = t * 2.0f - 3.0f;
  uint f = uint(face);
  float3 d;
  d.x = f == 0u ? 1.0f : f == 1u ? -1.0f : f == 5u ? -u : u;
  d.y = f == 2u ? 1.0f : f == 3u ? -1.0f : -v;
  d.z = f == 0u ? -u : f == 1u ? u : f == 2u ? v : f == 3u ? -v : f == 4u ? 1.0f : -1.0f;
  return d;
}
// Float class test (v_cmp_class_f32): bit k of mask tests class k.
bool hw_class(uint x, uint mask) {
  uint e = (x >> 23) & 0xffu, m = x & 0x7fffffu;
  uint t = e == 0xffu ? 9u : (e == 0u ? (m == 0u ? 6u : 7u) : 8u);
  if ((x >> 31) != 0u) t = 11u - t;
  if (e == 0xffu && m != 0u) t = (x & 0x400000u) != 0u ? 1u : 0u;
  return ((mask >> t) & 1u) != 0u;
}
// GFX10 buffer formats: channel conversion of a field of w bits.
uint hw_unorm(uint f, uint w) { return asuint(float(f) / float((1u << w) - 1u)); }
uint hw_snorm(uint f, uint w) {
  int s = int(f << (32u - w)) >> (32u - w);
  return asuint(max(float(s) / float((1u << (w - 1u)) - 1u), -1.0f));
}
uint hw_sext(uint f, uint w) { return w >= 32u ? f : uint(int(f << (32u - w)) >> (32u - w)); }
