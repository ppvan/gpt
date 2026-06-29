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

Let $H = 3$ (3 hidden units). We use tiny made-up weights to keep the math readable.

== Setup

$
x_t = mat(1) quad upright("shape") (1,1) quad arrow.l upright("Fibonacci input")
$

$
W_x = mat(0.5; 0.3; 0.1) quad upright("shape") (3,1)
$

$
W_h = mat(0.1, 0.0, 0.2; 0.0, 0.2, 0.1; 0.3, 0.1, 0.0) quad upright("shape") (3,3)
$

$
b_h = mat(0.1; 0.1; 0.1) quad upright("shape") (3,1)
$

$
h_(t-1) = mat(0.0; 0.0; 0.0) quad upright("shape") (3,1) quad arrow.l upright("zero at start")
$

$
W_y = mat(0.6, 0.4, 0.3) quad upright("shape") (1,3)
$

$
b_y = mat(0.0) quad upright("shape") (1,1)
$

== Step 1 — Compute $W_x dot x_t$ #h(0.5em) → shape $(3,1)$

$
mat(0.5; 0.3; 0.1) times mat(1) = mat(0.5; 0.3; 0.1)
$

== Step 2 — Compute $W_h dot h_(t-1)$ #h(0.5em) → shape $(3,1)$

$
mat(0.1, 0.0, 0.2; 0.0, 0.2, 0.1; 0.3, 0.1, 0.0)
times
mat(0.0; 0.0; 0.0)
=
mat(0.0; 0.0; 0.0)
$

(zero because $h$ is zero at $t=0$)

== Step 3 — Add everything + bias #h(0.5em) → shape $(3,1)$

$
mat(0.5; 0.3; 0.1) + mat(0.0; 0.0; 0.0) + mat(0.1; 0.1; 0.1) = mat(0.6; 0.4; 0.2)
$

== Step 4 — Apply $tanh$ #h(0.5em) → $h_t$, shape $(3,1)$

$
h_t = tanh mat(0.6; 0.4; 0.2) = mat(0.537; 0.380; 0.197)
$

== Step 5 — Compute output $y_t = W_y dot h_t + b_y$ #h(0.5em) → shape $(1,1)$

$
mat(0.6, 0.4, 0.3) times mat(0.537; 0.380; 0.197) + mat(0.0)
$

$
= mat(0.6 times 0.537 + 0.4 times 0.380 + 0.3 times 0.197) + 0
$

$
= mat(0.322 + 0.152 + 0.059)
$

$
= mat(0.533)
$

*Prediction:* $0.533$ — target was $1$. The network is untrained, so it's way off.
After backprop across many steps of $[1, 1, 2, 3, 5, 8, dots]$,
the weights adjust until predictions converge.