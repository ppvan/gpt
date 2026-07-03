#import "@preview/cetz:0.5.2"

== Recurrent Neural Network

=== Tại sao lại có RNN?

Tại thời điểm người ta nghĩ ra Deep Neural Network (DNN), nó xử lý rất tốt các dữ liệu "độc lập". Ví dụ nhận diện "mèo" trong tấm ảnh. Việc tấm ảnh 1 có mèo chả liên quan gì tới tấm ảnh 2, vì vậy DNN làm tốt nhiệm vụ này.

Với dữ liệu dạng sequence, DNN k hiệu quả vì nó không được thiết kế để làm vậy. Ví dụ, mô hình dự đoán từ tiếp theo:

"Hôm nay trời sẽ có ..."

#sym.arrow "mưa"

mô hình cần viết tất cả các từ phía trước để đưa ra dự đoán chính xác hơn. Nó cần nhớ được từ phía trước là gì, từ nào quan trọng...

Vậy nó cần 1 "bộ nhớ", bộ nhớ này cần được cập nhật thông tin khi có từ mới được nạp vào model

$
  "Memory" &= "SomeFunction" ("new_word", "old memory") \
  "memory"_t &= f("memory"_{t-1}, "input"_t) \
  h_t &= f(x_t, h_{t-1})
$

Đây chính là công thức Recurrent Neural Network. Vẽ thành diagram thì nó thành cái neuron xoay vòng ý.

#cetz.canvas({
  import cetz.draw: *

  // Styles
  set-style(
    circle: (radius: 0.5, fill: rgb("#eef4ff"), stroke: black),
    rect: (fill: rgb("#fff3e0"), stroke: black),
    content: (padding: 0.1)
  )

  // Input x_t
  rect((-4, -0.5), (-2.8, 0.5), name: "x")
  content("x.center", $x_t$)

  // RNN cell (tanh block)
  circle((0, 0), radius: 1, name: "cell")
  content("cell.center", $tanh$)

  // Output / hidden state h_t
  rect((2.8, -0.5), (4, 0.5), name: "h")
  content("h.center", $h_t$)

  // Weight labels
  content((-1.5, 0.6), $W_x$)
  content((1.5, 0.6), $W_h$)

  // Arrows: input -> cell
  line("x.east", "cell.west", mark: (end: ">"))

  // Arrows: cell -> output
  line("cell.east", "h.west", mark: (end: ">"))

  // Recurrent loop: h_t feeds back into cell (h_{t-1})
  line("h.north", (4, 2), (0, 2), "cell.north",
       mark: (end: ">"))
  content((2, 2.3), $h_{t-1}$)

  // Bias term
  circle((0, -2), radius: 0.35, name: "bias")
  content("bias.center", $b$)
  line("bias.north", "cell.south", mark: (end: ">"))
})

== Fomula

$
  h_t &= tanh(W_h · h_{t-1}  +  W_x · x_t  +  b_h) \
  y_t &= W_y · h_t  +  b_y
$

=== Matrix Dimensions
$
&"input_size" = 1 "(one number at a time)" \
&"hidden_size" = H "(your choice, e.g. 4)" \
&"output_size" = 1 "(predict one next number)" \
$

#table(
  columns: (auto, auto, 1fr),
  align: (center, center, left),
  table.header(
    [*Term*], [*Shape*], [*Description*],
  ),
  [$x_t$],   [$(1, 1)$],   [one Fibonacci number],
  [$h_t$],   [$(H, 1)$],   [hidden state vector],
  [$W_x$],   [$(H, 1)$],   [input weights],
  [$W_h$],   [$(H, H)$],   [hidden-to-hidden weights],
  [$b_h$],   [$(H, 1)$],   [hidden bias],
  [$W_y$],   [$(1, H)$],   [output weights],
  [$b_y$],   [$(1, 1)$],   [output bias],
)

= Step-by-step Example

Problem: Predict the Fibonacci sequence:
1, 1, 2, 3, 5 ....

Input: 1

Output: 2

Thêm cái RRN đơn giản nhất có thể:

- Input state: I = 1
- Hidden state: H = 3
- Output state: O = 1

$
  h_t &= tanh(W_h · h_{t-1}  +  W_x · x_t  +  b_h) \
  y_t &= W_y · h_t  +  b_y
$

Vì 

$
y_t = (1 times 1) => b_y = (1 times 1) \
h_t, h_(t-1) = (3 times 1) => W_y = (1 times 3) \

h_t, h_(t-1) = (3 times 1) => W_h = (3 times 3), W_x = (3 times 1), b_h = (3 times 1)
$


Loss:

$
  L = 1/ 2 sum_(t=0)^T (y^hat_t - y_t)^2\

  T = 4
$

#table(
  columns: (1fr, 1fr),
  align: left,

  [*Forward*], [*Backward*],

  [
  $
    z_t = W_x x_t + W_h h_(t-1) + b_h
  $
  ],
  [
  $
    (partial L)/(partial z_t)
      =
      (partial L)/(partial h_t)
      dot.o
      (1 - h_t^2)
  $
  ],

  [
  $
    h_t = tanh(z_t)
  $
  ],
  [
  $
    (partial L)/(partial h_t)
      =
      W_y^T
      (partial L)/(partial y^hat_t)
      +
      (partial L)/(partial h_(t+1))
  $
  ],

  [
  $
    y^hat_t = W_y h_t + b_y
  $
  ],
  [
  $
    (partial L)/(partial y^hat_t)
      =
      y^hat_t - y_t
  $
  ],

  [
  $
    W_y
  $
  ],
  [
  $
    (partial L)/(partial W_y)
      =
      (partial L)/(partial y^hat_t)
      h_t^T
  $
  ],

  [
  $
    b_y
  $
  ],
  [
  $
    (partial L)/(partial b_y)
      =
      (partial L)/(partial y^hat_t)
  $
  ],

  [
  $
    W_x
  $
  ],
  [
  $
    (partial L)/(partial W_x)
      =
      (partial L)/(partial z_t)
      x_t^T
  $
  ],

  [
  $
    W_h
  $
  ],
  [
  $
    (partial L)/(partial W_h)
      =
      (partial L)/(partial z_t)
      h_(t-1)^T
  $
  ],

  [
  $
    b_h
  $
  ],
  [
  $
    (partial L)/(partial b_h)
      =
      (partial L)/(partial z_t)
  $
  ],

  [
  $
    h_t
      arrow
      h_(t+1)
  $
  ],
  [
  $
    (partial L)/(partial h_(t-1))
      =
      W_h^T
      (partial L)/(partial z_t)
  $
  ],
)