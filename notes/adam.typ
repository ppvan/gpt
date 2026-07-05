#import "@preview/cetz:0.5.2"

== Adam optimizer

Trước khi tiếp cận Adam optimizer, ta cần hiểu hai phiên bản đơn giản hơn là: Momentumm và RMSProp


=== Momentum

Thay vì luôn update weights dựa theo gradient hiện tại, Momentum sử dụng 2 yếu tố: gradient hiện tại và trung bình những gradient trước đó.

Tương tự với normalization, Momentum có tác dụng "cancel out" những lần gradient di chuyển zig-zag (trung bình của zig-zag chính là 1 đường thẳng tối ưu).

#align(center)[
#cetz.canvas({
  import cetz.draw: *

  // ---- Các đường contour (loss landscape) hình elip lồng nhau ----
  let contour-color = rgb("#c9d6e8")
  for r in (5.5, 4.5, 3.5, 2.5, 1.5, 0.7) {
    circle((0, 0), radius: (r * 1.6, r), stroke: contour-color + 1pt)
  }
  content((0, 0), text(fill: rgb("#8aa0c2"), size: 9pt)[Điểm tối ưu])
  circle((0, 0), radius: 0.06, fill: rgb("#8aa0c2"), stroke: none)

  // ---- Đường đi zig-zag của Gradient Descent thường (không momentum) ----
  let zigzag = (
    (-7.5, 3.2), (-6.2, -2.0), (-5.1, 1.6), (-4.1, -1.1),
    (-3.3, 0.7), (-2.6, -0.5), (-2.0, 0.3), (-1.5, -0.15),
    (-1.1, 0.1), (-0.8, -0.05), (-0.5, 0), (0, 0)
  )
  for i in range(zigzag.len() - 1) {
    line(zigzag.at(i), zigzag.at(i + 1),
      stroke: rgb("#e0574c") + 1.6pt,
      mark: (end: ">", scale: 1.2, fill: rgb("#e0574c")))
  }

  // ---- Đường đi có Momentum: mượt, gần thẳng ----
  let momentum-path = (
    (-7.5, 3.2), (-6.0, 1.6), (-4.6, 0.55), (-3.4, -0.05),
    (-2.4, -0.15), (-1.5, -0.05), (-0.7, 0.02), (0, 0)
  )
  for i in range(momentum-path.len() - 1) {
    line(momentum-path.at(i), momentum-path.at(i + 1),
      stroke: rgb("#2c8c5c") + 2.4pt,
      mark: (end: ">", scale: 1.4, fill: rgb("#2c8c5c")))
  }

  // ---- Điểm bắt đầu ----
  circle((-7.5, 3.2), radius: 0.07, fill: black, stroke: none)
  content((-7.5, 3.7), text(size: 9pt)[Điểm bắt đầu])

  // ---- Chú thích (legend) ----
  line((2.0, 4.6), (2.9, 4.6), stroke: rgb("#e0574c") + 1.6pt)
  content((5.0, 4.6), text(size: 9pt, fill: rgb("#e0574c"))[Gradient Descent (zig-zag)])

  line((2.0, 4.0), (2.9, 4.0), stroke: rgb("#2c8c5c") + 2.4pt)
  content((4.4, 4.0), text(size: 9pt, fill: rgb("#2c8c5c"))[Có Momentum (mượt hơn)])
})
]

Trực giác: các bước gradient liên tiếp dao động qua lại hai bên "thung lũng" (zig-zag). Khi lấy trung bình động (moving average) của các gradient đó, các thành phần dao động ngang triệt tiêu lẫn nhau, chỉ còn lại thành phần hướng thẳng về điểm tối ưu — giúp hội tụ nhanh và ổn định hơn.
