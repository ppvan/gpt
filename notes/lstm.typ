=== LSTM

==== LSTM là gì?

=> Một kiến trúc mở rộng từ RNN, với mục đích giảm thiểu vấn đề vanishing gradient và expode gradient.
Do giữ được gradient lâu hơn, LSTM tự nhiên sinh ra khả năng lưu long term dependecies tốt hơn nhiều so với RNN. Ví dụ, RNN nhớ đc context 256 thì LSTM nhớ được context 1024.

=== Vì sao vanila RNN gặp vấn đề?

Giả sử ta có hidden state $h_t$ ở time step $t$. 
$
  h_t = sigma(w h_(t-1))
$

Giả sử ta cần tính đạo hàm của $h_t'$ trong đó $t'$ là thời điểm sau $t$.
Ví dụ $t' = t + 3$

$
  frac(partial h_(t+3), partial h_t) &= frac(partial h_(t+3), partial h_(t+2)) times frac(partial h_(t+2), partial h_(t+1)) times frac(partial h_(t+1), partial h_(t))
$

Xét riêng
$
  frac(partial h_(t+1), partial h_(t)) &= sigma (w h_t) \
    &= sigma'(w h_t).w
$


Tổng quát hoá


$
  frac(partial h_t', partial h_t) &= product^(t' - t)_(k=1) w sigma'(w h_(t-k)) \
  &= w^(t -t') product^(t' - t)_(k=1) sigma'(w h_(t-k)) \
$

Chú ý toán hạng: $w^(t-t')$ là luỹ thừa của w nên scale theo cấp luỹ thừa. Nếu $w < 1$ thì sau 100 step gradient vanish về $0$. Nếu $w > 1$ thì gradient expode.

#sym.arrow.double Ảnh hưởng của $h_1$ tới $h_(100)$ gần như bằng 0 hoặc $inf$, tức RNN rất khó học sự liên quan giữa các timestep dài

*Note*: đây là trường hợp đơn giản hoá với số thực, tổng quát với ma trận sẽ khác 1 chút nhưng idea thì tương tự

=== Idea

LSTM (Long Short-Term Memory) mở rộng RNN bằng cách bổ sung cell state $c_t$, đóng vai trò như bộ nhớ dài hạn, trong khi hidden state $h_t$ hoạt động như bộ nhớ ngắn hạn và là thông tin được truyền ra ngoài.

Một hệ thống các gate học cách điều khiển luồng thông tin: 
- Forget gate quyết định thông tin nào trong bộ nhớ dài hạn nên bị loại bỏ
- Input gate quyết định thông tin mới nào sẽ được ghi vào cell state
- Output gate quyết định phần nào của cell state sẽ được xuất thành hidden state.

#quote([Forget gate: nên quên đi cái gì, Input gate: nên nhớ cái gì, Output gate: nên nói gì
])

Nhờ cell state có một đường truyền gần như tuyến tính qua thời gian, LSTM cho phép gradient lan truyền ổn định hơn qua nhiều timestep, từ đó giảm đáng kể hiện tượng vanishing gradient so với RNN thông thường.

Khả năng này không phải ngẫu nhiên mà là một đặc tính được thiết kế của LSTM. Các công thức cập nhật cell state tạo ra một đường truyền gần như tuyến tính theo thời gian, giúp gradient có thể được duy trì gần 1 giữa các cell state liên tiếp. Điều này sẽ được làm rõ ở phần triển khai công thức.

=== Công thức toán học

Blog này visualize rất đỉnh về các gate: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

#image("image.png")

Từ hình ảnh ta thấy, từ input $x_(t)$ và hidden state $h_(t-1)$, ta cần tính ra $C_t$ và $h_t$

$C_t$ là đường thẳng ở phía trên của cell trong ảnh, nó cho phép thông tin đi từ $h_(t-1)$ qua $h_t$ mà không cần đi qua activation $tanh$, vì vậy thông tin có thể được truyền nguyên vẹn nếu Forget gate gần bằng 1 (có thể học đc)

Từ trái qua phải lần lượt là:
- Forget gate
- Input gate
- Output gate

==== Forget gate

Forget gate là một feed forward network thường
$
  f_t = sigma (W_f x_t + U_f h_(t-1) + b_f)
$

nó output 1 vector giữa 0 và 1 để quyết định, state nào nên nhớ, state nào nên quên trước khi ghi vào cell state.

==== Input gate

Input gate gồm 2 phần: 1 sigmoid và 1 tanh. Sigmoid đại diện cho việc nên nhờ bao nhiều % của thông tin, còn tanh chính là thông tin đầu vào (tương tự tanh trong vanila RNN)

$
  i_t &= sigma(W_i x_t + U_i h_(t-1) + b_i) \
  c^~_t &= tanh (W_c x_t + U_c h_(t-1) + b_c))
$

Cập nhật bộ nhớ:

$
  c_t = f_t dot.o c_(t-1) + i_t dot.o c^~_t
$

với f_t chính là value của forget gate phía trên

==== Output gate

$
o_t = sigma (W_o x_t + U_o h_(t-1) + b_o) \
h_t = o_t dot.o tanh(c_t)
$
== LSTM Backpropagation

Tại thời điểm bắt đầu của time step $t$, ta đã có:

$
  delta h_t "từ loss hoặc timestep sau" \
  delta c_(t+1) "từ timestep sau"
$

Mục tiêu là tính:

- Gradient của toàn bộ tham số $(W_*, U_*, b_*)$
- $delta h_(t-1)$
- $delta c_(t-1)$

sau đó tiếp tục đệ quy sang timestep trước.

== Bước 1: Tính $delta o_t$

Trong bước forward:

$
h_t = o_t dot.o tanh(c_t)
$

Khi lấy đạo hàm theo $o_t$, $tanh(c_t)$ được xem là hằng số.

$
delta o_t
=
delta h_t dot.o tanh(c_t)
$

== Bước 2: Tính $delta c_t$

Trong forward:

$
c_t
=
f_t dot.o c_(t-1)
+
i_t dot.o g_t
$

$
h_t
=
o_t dot.o tanh(c_t)
$

Gradient của $c_t$ đến từ hai hướng:

- Cell state của timestep sau
- Hidden state hiện tại

$
delta c_t
=
delta c_(t+1) dot.o f_(t+1)
+
delta h_t dot.o o_t dot.o (1 - tanh^2(c_t))
$

== Bước 3: Tính gradient của các gate

Từ

$
c_t
=
f_t dot.o c_(t-1)
+
i_t dot.o g_t
$

suy ra

$
delta f_t
=
delta c_t dot.o c_(t-1)
$

$
delta c_(t-1)
=
delta c_t dot.o f_t
$

$
delta i_t
=
delta c_t dot.o g_t
$

$
delta g_t
=
delta c_t dot.o i_t
$

== Bước 4: Backprop qua activation

Forward:

$
f_t = sigma(z_f) \
i_t = sigma(z_i) \
o_t = sigma(z_o) \
g_t = tanh(z_g)
$

Gradient theo pre-activation:

$
delta z_f
=
delta f_t dot.o f_t dot.o (1 - f_t)
$

$
delta z_i
=
delta i_t dot.o i_t dot.o (1 - i_t)
$

$
delta z_o
=
delta o_t dot.o o_t dot.o (1 - o_t)
$

$
delta z_g
=
delta g_t dot.o (1 - g_t^2)
$

== Bước 5: Tính gradient của các tham số

Forward:

$
z_*
=
x_t W_*
+
h_(t-1) U_*
+
b_*
$

Do sử dụng row vector notation:

$
delta W_*
+=
x_t^T delta z_*
$

$
delta U_*
+=
h_(t-1)^T delta z_*
$

$
delta b_*
+=
delta z_*
$

với

$
* in {f, i, o, g}
$

== Bước 6: Tính $delta h_(t-1)$

Do $h_(t-1)$ ảnh hưởng tới cả bốn gate:

$
delta h_(t-1)
=
delta z_f U_f^T
+
delta z_i U_i^T
+
delta z_o U_o^T
+
delta z_g U_g^T
$

Sau khi hoàn thành timestep $t$, ta truyền hai gradient sang timestep trước:

$
delta h_(t-1), quad
delta c_(t-1)
$

và tiếp tục backpropagation.

=== Coding

