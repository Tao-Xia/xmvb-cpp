#include "vb/pdft/lebedev_grid.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace xmvb::vb::pdft {

namespace {

constexpr double kFourPi = 4.0 * M_PI;

void add_point(
    double x,
    double y,
    double z,
    double weight,
    std::vector<LebedevPoint>* points) {
  points->push_back({x, y, z, weight});
}

/**
 * @brief Adds the 6-point octahedral shell from legacy `add_point_1`.
 */
void add_octahedral_a_points(
    double weight,
    std::vector<LebedevPoint>* points) {
  const double scaled_weight = kFourPi * weight;
  add_point(1.0, 0.0, 0.0, scaled_weight, points);
  add_point(-1.0, 0.0, 0.0, scaled_weight, points);
  add_point(0.0, 1.0, 0.0, scaled_weight, points);
  add_point(0.0, -1.0, 0.0, scaled_weight, points);
  add_point(0.0, 0.0, 1.0, scaled_weight, points);
  add_point(0.0, 0.0, -1.0, scaled_weight, points);
}

/**
 * @brief Adds the 12-point octahedral shell from legacy `add_point_2`.
 */
void add_octahedral_b_points(
    double a,
    double weight,
    std::vector<LebedevPoint>* points) {
  const double scaled_weight = kFourPi * weight;
  const double b = std::sqrt(1.0 - a * a);
  add_point(0.0, a, b, scaled_weight, points);
  add_point(0.0, -a, b, scaled_weight, points);
  add_point(0.0, a, -b, scaled_weight, points);
  add_point(0.0, -a, -b, scaled_weight, points);
  add_point(a, 0.0, b, scaled_weight, points);
  add_point(-a, 0.0, b, scaled_weight, points);
  add_point(a, 0.0, -b, scaled_weight, points);
  add_point(-a, 0.0, -b, scaled_weight, points);
  add_point(a, b, 0.0, scaled_weight, points);
  add_point(-a, b, 0.0, scaled_weight, points);
  add_point(a, -b, 0.0, scaled_weight, points);
  add_point(-a, -b, 0.0, scaled_weight, points);
}

/**
 * @brief Adds the 8-point octahedral shell from legacy `add_point_3`.
 */
void add_octahedral_c_points(
    double a,
    double weight,
    std::vector<LebedevPoint>* points) {
  const double scaled_weight = kFourPi * weight;
  for (int sx = -1; sx <= 1; sx += 2) {
    for (int sy = -1; sy <= 1; sy += 2) {
      for (int sz = -1; sz <= 1; sz += 2) {
        add_point(sx * a, sy * a, sz * a, scaled_weight, points);
      }
    }
  }
}

/**
 * @brief Adds the 24-point shell from legacy `add_point_4`.
 *
 * The generated coordinates are permutations of `(±a, ±a, ±b)` with
 * `b = sqrt(1 - 2a^2)`.
 */
void add_octahedral_d_points(
    double a,
    double weight,
    std::vector<LebedevPoint>* points) {
  const double scaled_weight = kFourPi * weight;
  const double b = std::sqrt(1.0 - 2.0 * a * a);
  add_point(a, a, b, scaled_weight, points);
  add_point(-a, a, b, scaled_weight, points);
  add_point(a, -a, b, scaled_weight, points);
  add_point(a, a, -b, scaled_weight, points);
  add_point(-a, -a, b, scaled_weight, points);
  add_point(-a, a, -b, scaled_weight, points);
  add_point(a, -a, -b, scaled_weight, points);
  add_point(-a, -a, -b, scaled_weight, points);
  add_point(a, b, a, scaled_weight, points);
  add_point(-a, b, a, scaled_weight, points);
  add_point(a, -b, a, scaled_weight, points);
  add_point(a, b, -a, scaled_weight, points);
  add_point(-a, -b, a, scaled_weight, points);
  add_point(-a, b, -a, scaled_weight, points);
  add_point(a, -b, -a, scaled_weight, points);
  add_point(-a, -b, -a, scaled_weight, points);
  add_point(b, a, a, scaled_weight, points);
  add_point(-b, a, a, scaled_weight, points);
  add_point(b, -a, a, scaled_weight, points);
  add_point(b, a, -a, scaled_weight, points);
  add_point(-b, -a, a, scaled_weight, points);
  add_point(-b, a, -a, scaled_weight, points);
  add_point(b, -a, -a, scaled_weight, points);
  add_point(-b, -a, -a, scaled_weight, points);
}

/**
 * @brief Adds the 24-point shell from legacy `add_point_5`.
 *
 * The generated coordinates are permutations of `(±a, ±b, 0)` with
 * `b = sqrt(1 - a^2)`.
 */
void add_octahedral_e_points(
    double a,
    double weight,
    std::vector<LebedevPoint>* points) {
  const double scaled_weight = kFourPi * weight;
  const double b = std::sqrt(1.0 - a * a);
  add_point(a, b, 0.0, scaled_weight, points);
  add_point(-a, b, 0.0, scaled_weight, points);
  add_point(a, -b, 0.0, scaled_weight, points);
  add_point(-a, -b, 0.0, scaled_weight, points);
  add_point(b, a, 0.0, scaled_weight, points);
  add_point(-b, a, 0.0, scaled_weight, points);
  add_point(b, -a, 0.0, scaled_weight, points);
  add_point(-b, -a, 0.0, scaled_weight, points);
  add_point(a, 0.0, b, scaled_weight, points);
  add_point(-a, 0.0, b, scaled_weight, points);
  add_point(a, 0.0, -b, scaled_weight, points);
  add_point(-a, 0.0, -b, scaled_weight, points);
  add_point(b, 0.0, a, scaled_weight, points);
  add_point(-b, 0.0, a, scaled_weight, points);
  add_point(b, 0.0, -a, scaled_weight, points);
  add_point(-b, 0.0, -a, scaled_weight, points);
  add_point(0.0, a, b, scaled_weight, points);
  add_point(0.0, -a, b, scaled_weight, points);
  add_point(0.0, a, -b, scaled_weight, points);
  add_point(0.0, -a, -b, scaled_weight, points);
  add_point(0.0, b, a, scaled_weight, points);
  add_point(0.0, -b, a, scaled_weight, points);
  add_point(0.0, b, -a, scaled_weight, points);
  add_point(0.0, -b, -a, scaled_weight, points);
}

/**
 * @brief Adds the 48-point shell from legacy `add_point_6`.
 */
void add_octahedral_f_points(
    double a,
    double b,
    double weight,
    std::vector<LebedevPoint>* points) {
  const double scaled_weight = kFourPi * weight;
  const double c = std::sqrt(1.0 - a * a - b * b);
  add_point(a, b, c, scaled_weight, points);
  add_point(-a, b, c, scaled_weight, points);
  add_point(a, -b, c, scaled_weight, points);
  add_point(a, b, -c, scaled_weight, points);
  add_point(-a, -b, c, scaled_weight, points);
  add_point(-a, b, -c, scaled_weight, points);
  add_point(a, -b, -c, scaled_weight, points);
  add_point(-a, -b, -c, scaled_weight, points);
  add_point(a, c, b, scaled_weight, points);
  add_point(-a, c, b, scaled_weight, points);
  add_point(a, -c, b, scaled_weight, points);
  add_point(a, c, -b, scaled_weight, points);
  add_point(-a, -c, b, scaled_weight, points);
  add_point(-a, c, -b, scaled_weight, points);
  add_point(a, -c, -b, scaled_weight, points);
  add_point(-a, -c, -b, scaled_weight, points);
  add_point(b, a, c, scaled_weight, points);
  add_point(-b, a, c, scaled_weight, points);
  add_point(b, -a, c, scaled_weight, points);
  add_point(b, a, -c, scaled_weight, points);
  add_point(-b, -a, c, scaled_weight, points);
  add_point(-b, a, -c, scaled_weight, points);
  add_point(b, -a, -c, scaled_weight, points);
  add_point(-b, -a, -c, scaled_weight, points);
  add_point(b, c, a, scaled_weight, points);
  add_point(-b, c, a, scaled_weight, points);
  add_point(b, -c, a, scaled_weight, points);
  add_point(b, c, -a, scaled_weight, points);
  add_point(-b, -c, a, scaled_weight, points);
  add_point(-b, c, -a, scaled_weight, points);
  add_point(b, -c, -a, scaled_weight, points);
  add_point(-b, -c, -a, scaled_weight, points);
  add_point(c, a, b, scaled_weight, points);
  add_point(-c, a, b, scaled_weight, points);
  add_point(c, -a, b, scaled_weight, points);
  add_point(c, a, -b, scaled_weight, points);
  add_point(-c, -a, b, scaled_weight, points);
  add_point(-c, a, -b, scaled_weight, points);
  add_point(c, -a, -b, scaled_weight, points);
  add_point(-c, -a, -b, scaled_weight, points);
  add_point(c, b, a, scaled_weight, points);
  add_point(-c, b, a, scaled_weight, points);
  add_point(c, -b, a, scaled_weight, points);
  add_point(c, b, -a, scaled_weight, points);
  add_point(-c, -b, a, scaled_weight, points);
  add_point(-c, b, -a, scaled_weight, points);
  add_point(c, -b, -a, scaled_weight, points);
  add_point(-c, -b, -a, scaled_weight, points);
}

// Lebedev grid generators for specific sizes

std::vector<LebedevPoint> generate_lebedev_6() {
  std::vector<LebedevPoint> points;
  points.reserve(6);
  add_octahedral_a_points(0.1666666666666667, &points);
  return points;
}

std::vector<LebedevPoint> generate_lebedev_14() {
  std::vector<LebedevPoint> points;
  points.reserve(14);
  add_octahedral_a_points(0.06666666666666667, &points);
  add_octahedral_c_points(0.5773502691896257, 0.07500000000000000, &points);
  return points;
}

std::vector<LebedevPoint> generate_lebedev_26() {
  std::vector<LebedevPoint> points;
  points.reserve(26);
  add_octahedral_a_points(0.04761904761904762, &points);
  add_octahedral_b_points(0.7071067811865475, 0.03809523809523810, &points);
  add_octahedral_c_points(0.5773502691896257, 0.03214285714285714, &points);
  return points;
}

std::vector<LebedevPoint> generate_lebedev_38() {
  std::vector<LebedevPoint> points;
  points.reserve(38);
  add_octahedral_a_points(0.009523809523809524, &points);
  add_octahedral_c_points(0.5773502691896257, 0.03214285714285714, &points);
  add_octahedral_e_points(0.4597008433809831, 0.02857142857142857, &points);
  return points;
}

std::vector<LebedevPoint> generate_lebedev_50() {
  std::vector<LebedevPoint> points;
  points.reserve(50);
  add_octahedral_a_points(0.01269841269841270, &points);
  add_octahedral_b_points(0.7071067811865475, 0.02257495590828924, &points);
  add_octahedral_c_points(0.5773502691896257, 0.02109375000000000, &points);
  add_octahedral_d_points(0.3015113445777636, 0.02017333553791887, &points);
  return points;
}

std::vector<LebedevPoint> generate_lebedev_302() {
  // Lebedev-302 grid copied from the established `xgrids.c` table.
  std::vector<LebedevPoint> points;
  points.reserve(302);

  add_octahedral_a_points(8.545911725128148e-4, &points);
  add_octahedral_c_points(5.773502691896258e-1, 3.599119285025571e-3, &points);
  add_octahedral_d_points(3.515640345570105e-1, 3.449788424305883e-3, &points);
  add_octahedral_d_points(6.566329410219612e-1, 3.604822601419882e-3, &points);
  add_octahedral_d_points(4.729054132581005e-1, 3.576729661743367e-3, &points);
  add_octahedral_d_points(9.618308522614784e-2, 2.352101413689164e-3, &points);
  add_octahedral_d_points(2.219645236294178e-1, 3.108953122413675e-3, &points);
  add_octahedral_d_points(7.011766416089545e-1, 3.650045807677255e-3, &points);
  add_octahedral_e_points(2.644152887060663e-1, 2.982344963171804e-3, &points);
  add_octahedral_e_points(5.718955891878961e-1, 3.600820932216460e-3, &points);
  add_octahedral_f_points(2.510034751770465e-1, 8.000727494073952e-1,
                          3.571540554273387e-3, &points);
  add_octahedral_f_points(1.233548532583327e-1, 4.127724083168531e-1,
                          3.392312205006170e-3, &points);

  return points;
}

}  // anonymous namespace

std::vector<LebedevPoint> LebedevGrid::get_grid(int n_points) {
  switch (n_points) {
    case 6:
      return generate_lebedev_6();
    case 14:
      return generate_lebedev_14();
    case 26:
      return generate_lebedev_26();
    case 38:
      return generate_lebedev_38();
    case 50:
      return generate_lebedev_50();
    case 302:
      return generate_lebedev_302();
    default: {
      throw std::invalid_argument("Unsupported Lebedev grid size");
    }
  }
}

std::vector<int> LebedevGrid::get_supported_sizes() {
  return {6, 14, 26, 38, 50, 302};
}

int LebedevGrid::get_closest_supported_size(int requested_size) {
  const auto sizes = get_supported_sizes();
  auto it = std::lower_bound(sizes.begin(), sizes.end(), requested_size);
  if (it == sizes.end()) {
    return sizes.back();
  }
  return *it;
}

bool LebedevGrid::is_supported(int n_points) {
  const auto sizes = get_supported_sizes();
  return std::find(sizes.begin(), sizes.end(), n_points) != sizes.end();
}

}  // namespace xmvb::vb::pdft
