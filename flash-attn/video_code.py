from manimlib import *

COMPUTE_W = 0.5
COMPUTE_H = 1.6
COMPUTE_R = 0.1

COMPUTE_COLOR = RED
COMMON_STROKE_WIDTH = 2

TEXT_BUFF = 0.5
TEXT_ROTATE_ANGLE = -90 * DEGREES

def create_tensor(dshape=(6, 1), color=GREEN_A, grid_width=0.3, grid_height=0.3):
    grid_group = VGroup()

    for i in range(dshape[0]):
        for j in range(dshape[1]):
            grid = Square(
                side_length=grid_width,
                fill_color=BLACK,
                stroke_width=COMMON_STROKE_WIDTH,
                stroke_color=color,
                fill_opacity=0.5
            )
            grid.move_to((i * grid_height * UP) + (j * grid_width * RIGHT))
            grid_group.add(grid)

    return grid_group

WAIT_TIME_SHORT = 0.01

class AttentionArch(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        
        inputs = create_tensor().shift(LEFT * 6.5 + DOWN * 0.5)
        inputs_label = Tex("x").scale(0.8).next_to(inputs, DOWN)
        self.play(
            Write(inputs_label),
            ShowCreation(inputs)
        )

        self.wait(WAIT_TIME_SHORT)
        k_proj = RoundedRectangle(
            width=COMPUTE_W,
            height=COMPUTE_H,
            fill_color=BLACK,
            corner_radius=COMPUTE_R,
            stroke_width=COMMON_STROKE_WIDTH,
            fill_opacity=0.5,
            stroke_color=COMPUTE_COLOR).next_to(inputs, RIGHT, buff=1)

        q_proj = RoundedRectangle(
            width=COMPUTE_W, 
            height=COMPUTE_H,
            fill_color=BLACK, 
            corner_radius=COMPUTE_R,
            stroke_width=COMMON_STROKE_WIDTH,
            fill_opacity=0.5, 
            stroke_color=COMPUTE_COLOR).next_to(k_proj, UP, buff=1)

        v_proj = RoundedRectangle(
            width=COMPUTE_W,
            height=COMPUTE_H,
            fill_color=BLACK,
            corner_radius=COMPUTE_R,
            stroke_width=COMMON_STROKE_WIDTH,
            fill_opacity=0.5,
            stroke_color=COMPUTE_COLOR).next_to(k_proj, DOWN, buff=1)

        q_proj_label = Text("Q-Proj").rotate(TEXT_ROTATE_ANGLE).scale(0.3).move_to(q_proj)
        k_proj_label = Text("K-Proj").rotate(TEXT_ROTATE_ANGLE).scale(0.3).move_to(k_proj)
        v_proj_label = Text("V-Proj").rotate(TEXT_ROTATE_ANGLE).scale(0.3).move_to(v_proj)

        self.play(
            ShowCreation(q_proj),
            ShowCreation(k_proj),
            ShowCreation(v_proj),
            Write(q_proj_label),
            Write(k_proj_label),
            Write(v_proj_label)
        )

        self.wait(WAIT_TIME_SHORT)

        # 画从inputs到q_proj， k_proj, v_proj的弯曲箭头
        q_arrow = Arrow(
            inputs.get_right(), q_proj.get_left(), 
            path_arc=-PI / 6, color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7)
        k_arrow = Arrow(
            inputs.get_right(), k_proj.get_left(),
             color=WHITE,buff=0.0, stroke_width=COMMON_STROKE_WIDTH* 1.7)
        v_arrow = Arrow(
            inputs.get_right(), v_proj.get_left(),
            path_arc=PI / 6, color=WHITE,buff=0.0, stroke_width=COMMON_STROKE_WIDTH* 1.7)
        
        tmp_q_inp, tmp_k_inp, tmp_v_inp = inputs.copy(), inputs.copy(), inputs.copy()
        
        self.play(
            ShowCreation(q_arrow),
            ShowCreation(k_arrow),
            ShowCreation(v_arrow),
            MoveAlongPath(tmp_q_inp, q_arrow),
            MoveAlongPath(tmp_k_inp, k_arrow),
            MoveAlongPath(tmp_v_inp, v_arrow)
        )
        self.play(
            FadeOut(tmp_q_inp),
            FadeOut(tmp_k_inp),
            FadeOut(tmp_v_inp),
            Indicate(q_proj),
            Indicate(k_proj),
            Indicate(v_proj)
        )

        vec_buffer = 0.5
        q_vec = create_tensor(color=GREEN_B).next_to(q_proj, RIGHT, buff=vec_buffer)
        k_vec = create_tensor(color=GREEN_C).next_to(k_proj, RIGHT, buff=vec_buffer)
        v_vec = create_tensor(color=GREEN_D).next_to(v_proj, RIGHT, buff=vec_buffer)

        q_vec_label = Tex("Q").scale(0.6).next_to(q_vec, DOWN)
        k_vec_label = Tex("K").scale(0.6).next_to(k_vec, DOWN)
        v_vec_label = Tex("V").scale(0.6).next_to(v_vec, DOWN)


        q_vec_arrow = Arrow(q_proj.get_right(), q_vec.get_left(), color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7)
        k_vec_arrow = Arrow(k_proj.get_right(), k_vec.get_left(), color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7)
        v_vec_arrow = Arrow(v_proj.get_right(), v_vec.get_left(), color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7)

        self.play(
            ShowCreation(q_vec_arrow),
            ShowCreation(k_vec_arrow),
            ShowCreation(v_vec_arrow)
        )

        self.play(
            Write(q_vec_label),
            Write(k_vec_label),
            Write(v_vec_label),
            ShowCreation(q_vec),
            ShowCreation(k_vec),
            ShowCreation(v_vec)
        )
        self.wait(WAIT_TIME_SHORT)

        q_kt_mul = RoundedRectangle(
            width=COMPUTE_W * 1.2,
            height=COMPUTE_H *1.5,
            fill_color=BLACK,
            corner_radius=COMPUTE_R,
            stroke_width=COMMON_STROKE_WIDTH,
            fill_opacity=0.5,
            stroke_color=COMPUTE_COLOR).move_to((q_vec.get_center() + k_vec.get_center()) / 2 + RIGHT * 1.2)
        q_kt_mul_label = Text("matmul").scale(0.4).rotate(TEXT_ROTATE_ANGLE).move_to(q_kt_mul)
        self.play(
            ShowCreation(q_kt_mul),
            Write(q_kt_mul_label)
        )
        self.wait(WAIT_TIME_SHORT)

        # 画从inputs到q_proj， k_proj, v_proj的弯曲箭头
        q_vec_arrow = Arrow(
            q_vec.get_right(), q_kt_mul.get_left(), 
            path_arc=-PI / 6, color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7)
        k_vec_arrow = Arrow(
            k_vec.get_right(), q_kt_mul.get_left(),
            path_arc=PI / 6, color=WHITE,buff=0.0, stroke_width=COMMON_STROKE_WIDTH* 1.7)

        tmp_q_vec, tmp_k_vec = q_vec.copy(), k_vec.copy()

        self.play(
            Rotate(tmp_k_vec, angle=PI / 2)
        )
        # tmp_k_vec.rotate(PI / 2)

        self.play(
            ShowCreation(q_vec_arrow),
            ShowCreation(k_vec_arrow),
            MoveAlongPath(tmp_q_vec, q_vec_arrow),
            MoveAlongPath(tmp_k_vec, k_vec_arrow)
        )

        self.play(
            FadeOut(tmp_q_vec),
            FadeOut(tmp_k_vec),
            Indicate(q_kt_mul)
        )

        q_kt = create_tensor(dshape=(6, 6), grid_width=0.2, grid_height=0.2, color=GREEN_B).next_to(q_kt_mul, RIGHT, buff=vec_buffer)
        q_kt_label = Tex("Q \cdot K^T").scale(0.6).next_to(q_kt, DOWN)

        q_kt_mul_arrow = Arrow(
            q_kt_mul.get_right(), q_kt.get_left(),
            color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7)

        self.play(
            ShowCreation(q_kt_mul_arrow)
        )

        # 显示矩阵
        self.play(
            Write(q_kt_label),
            ShowCreation(q_kt)
        )
        self.wait(WAIT_TIME_SHORT)

        scale_op = RoundedRectangle(
            width=COMPUTE_W,
            height=COMPUTE_H,
            fill_color=BLACK,
            corner_radius=COMPUTE_R,
            stroke_width=COMMON_STROKE_WIDTH,
            fill_opacity=0.5,
            stroke_color=COMPUTE_COLOR).next_to(q_kt, RIGHT, buff=0.5)
        mask_op = RoundedRectangle(
            width=COMPUTE_W,
            height=COMPUTE_H,
            fill_color=BLACK,
            corner_radius=COMPUTE_R,
            stroke_width=COMMON_STROKE_WIDTH,
            fill_opacity=0.5,
            stroke_color=COMPUTE_COLOR).next_to(scale_op, RIGHT, buff=0.5)
        softmax_op = RoundedRectangle(
            width=COMPUTE_W,
            height=COMPUTE_H,
            fill_color=BLACK,
            corner_radius=COMPUTE_R,
            stroke_width=COMMON_STROKE_WIDTH,
            fill_opacity=0.5,
            stroke_color=COMPUTE_COLOR).next_to(mask_op, RIGHT, buff=0.5)
        dropout_op = RoundedRectangle(
            width=COMPUTE_W,
            height=COMPUTE_H,
            fill_color=BLACK,
            corner_radius=COMPUTE_R,
            stroke_width=COMMON_STROKE_WIDTH,
            fill_opacity=0.5,
            stroke_color=COMPUTE_COLOR).next_to(softmax_op, RIGHT, buff=0.5)
        

        scale_op_label = Text("Scale").scale(0.3).rotate(TEXT_ROTATE_ANGLE).move_to(scale_op)
        mask_op_label = Text("Mask(Opt)").scale(0.3).rotate(TEXT_ROTATE_ANGLE).move_to(mask_op)
        softmax_op_label = Text("Softmax").scale(0.3).rotate(TEXT_ROTATE_ANGLE).move_to(softmax_op)
        dropout_op_label = Text("Dropout").scale(0.3).rotate(TEXT_ROTATE_ANGLE).move_to(dropout_op)

        scale_op_arrow = Arrow(
            q_kt.get_right(), scale_op.get_left(),
            color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7)
        mask_op_arrow = Arrow(
            scale_op.get_right(), mask_op.get_left(),
            color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7)
        softmax_op_arrow = Arrow(
            mask_op.get_right(), softmax_op.get_left(),
            color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7)
        dropout_op_arrow = Arrow(
            softmax_op.get_right(), dropout_op.get_left(),
            color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7)
        
                
        tmp_q_kt = q_kt.copy()
        self.play(
            ShowCreation(scale_op),
            Write(scale_op_label)
        )
        self.play(
            ShowCreation(scale_op_arrow),
            MoveAlongPath(tmp_q_kt, scale_op_arrow)
        )
        self.wait(WAIT_TIME_SHORT)

        self.play(
            Indicate(scale_op),
            ShowCreation(mask_op),
            Write(mask_op_label))
        
        self.play(
            ShowCreation(mask_op_arrow),
            MoveAlongPath(tmp_q_kt, mask_op_arrow)
        )
        self.wait(WAIT_TIME_SHORT)

        self.play(
            Indicate(mask_op),
            ShowCreation(softmax_op),
            Write(softmax_op_label)
        )

        self.play(
            ShowCreation(softmax_op_arrow),
            MoveAlongPath(tmp_q_kt, softmax_op_arrow)
        )
        self.wait(WAIT_TIME_SHORT)

        self.play(
            Indicate(softmax_op),
            ShowCreation(dropout_op),
            Write(dropout_op_label)
        )

        self.play(
            ShowCreation(dropout_op_arrow),
            MoveAlongPath(tmp_q_kt, dropout_op_arrow)
        )
        self.wait(WAIT_TIME_SHORT)

        self.play(
            Indicate(dropout_op)
        )

        qktv_mul = RoundedRectangle(
            width=COMPUTE_W * 1.2,
            height=COMPUTE_H *1.5,
            fill_color=BLACK,
            corner_radius=COMPUTE_R,
            stroke_width=COMMON_STROKE_WIDTH,
            fill_opacity=0.5,
            stroke_color=COMPUTE_COLOR).move_to((dropout_op.get_x() + 1.2, k_proj.get_y(), 0))
        qktv_mul_label = Text("matmul").scale(0.4).rotate(TEXT_ROTATE_ANGLE).move_to(qktv_mul)
        self.play(
            ShowCreation(qktv_mul),
            Write(qktv_mul_label)
        )
        self.wait(WAIT_TIME_SHORT)

        qktv_mul_arrow_qkt = Arrow(
            dropout_op.get_right(), qktv_mul.get_left(),
            path_arc=-PI/4, color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7)
        qktv_mul_arrow_v = Arrow(
            v_vec.get_right(), qktv_mul.get_left(),
            path_arc=PI/6, color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7)
        
        tmp_v_vec = v_vec.copy()
        self.play(
            ShowCreation(qktv_mul_arrow_qkt),
            ShowCreation(qktv_mul_arrow_v),
            MoveAlongPath(tmp_q_kt, qktv_mul_arrow_qkt),
            MoveAlongPath(tmp_v_vec, qktv_mul_arrow_v)
        )

        self.play(
            FadeOut(tmp_q_kt),
            FadeOut(tmp_v_vec),
            Indicate(qktv_mul)
        )

        o = create_tensor(color=GREEN_E).next_to(qktv_mul, RIGHT, buff=vec_buffer)
        o_label = Tex("O").scale(0.6).next_to(o, DOWN)

        qktv_mul_arrow_out = Arrow(
            qktv_mul.get_right(), o.get_left(),
            color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7)
        
        self.play(
            ShowCreation(qktv_mul_arrow_out),
        )
        self.play(
            Write(o_label),
            ShowCreation(o)
        )
        self.wait(WAIT_TIME_SHORT)

        o_proj = RoundedRectangle(
            width=COMPUTE_W,
            height=COMPUTE_H,
            fill_color=BLACK,
            corner_radius=COMPUTE_R,
            stroke_width=COMMON_STROKE_WIDTH,
            fill_opacity=0.5,
            stroke_color=COMPUTE_COLOR).next_to(o, RIGHT, buff=0.5)
        o_proj_label = Text("O-Proj").scale(0.3).rotate(TEXT_ROTATE_ANGLE).move_to(o_proj)

        o_proj_arrow = Arrow(
            o.get_right(), o_proj.get_left(),
            color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7)
        
        tmp_o = o.copy()

        self.play(
            ShowCreation(o_proj),
            Write(o_proj_label),
        )
        self.wait(WAIT_TIME_SHORT)

        self.play(
            ShowCreation(o_proj_arrow),
            MoveAlongPath(tmp_o, o_proj_arrow)
        )

        outputs = create_tensor(color=GREEN_SCREEN).next_to(o_proj, RIGHT, buff=0.5)
        outputs_label = Tex("y").scale(0.8).next_to(outputs, DOWN)

        outputs_arrow = Arrow(
            o_proj.get_right(), outputs.get_left(),
            color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7)
        
        self.play(
            Indicate(o_proj),
            FadeOut(tmp_o),
            ShowCreation(outputs_arrow),
        )
        self.wait(WAIT_TIME_SHORT)

        self.play(
            Write(outputs_label),
            ShowCreation(outputs)
        )

        self.wait(WAIT_TIME_SHORT)

        fused_att = RoundedRectangle(
            width=COMPUTE_W * 2,
            height=COMPUTE_H * 2,
            fill_color=BLACK,
            corner_radius=COMPUTE_R,
            stroke_width=COMMON_STROKE_WIDTH * 1.5,
            fill_opacity=0.5,
            stroke_color=COMPUTE_COLOR).move_to((k_vec.get_center() + o.get_center()) / 2)
        fused_att_label = Text("FlashAttention").scale(0.4).rotate(TEXT_ROTATE_ANGLE).move_to(fused_att)

        new_q_flash_arrow = Arrow(
            q_vec.get_right(), fused_att.get_left(),
            path_arc= -PI/6, color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7
        )

        new_k_flash_arrow = Arrow(
            k_vec.get_right(), fused_att.get_left(),
            color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7
        )

        new_v_flash_arrow = Arrow(
            v_vec.get_right(), fused_att.get_left(),
            path_arc= PI/6, color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7
        )

        new_o_flash_arrow = Arrow(
             fused_att.get_right(), o.get_left(),
            color=WHITE, buff=0.0, stroke_width=COMMON_STROKE_WIDTH * 1.7
        )

        transform_inputs = [
            q_kt,
            q_kt_mul,
            scale_op,
            mask_op, 
            softmax_op,
            dropout_op,
            qktv_mul
        ]
        label_arrows_fadeout = [
            q_kt_label,
            q_kt_mul_label,
            scale_op_label,
            mask_op_label,
            softmax_op_label,
            dropout_op_label,
            qktv_mul_label,
            q_kt_mul_arrow,
            scale_op_arrow, 
            mask_op_arrow, 
            softmax_op_arrow, 
            dropout_op_arrow,
            qktv_mul_arrow_qkt
        ]
        self.play(
            ShowCreation(fused_att),
            Write(fused_att_label),
            *[FadeOutToPoint(obj, fused_att.get_center()) for obj in label_arrows_fadeout],
            *[Transform(obj, fused_att) for obj in transform_inputs],
            Transform(q_vec_arrow, new_q_flash_arrow),
            Transform(k_vec_arrow, new_k_flash_arrow),
            Transform(qktv_mul_arrow_v, new_v_flash_arrow),
            Transform(qktv_mul_arrow_out, new_o_flash_arrow),
        )
        self.wait(WAIT_TIME_SHORT)
        
        tmp_q_vec, tmp_k_vec, tmp_v_vec = q_vec.copy(), k_vec.copy(), v_vec.copy()

        o_color = [YELLOW_E, YELLOW_A, GREEN]

        for _, step_q in enumerate([4,2,0]):
            s_q_vec_outside = tmp_q_vec[step_q:step_q + 2]
            s_q_vec_outside.set_color(YELLOW_E)
            s_q_vec_outside.set_depth(-1)
            self.play(ShowCreation(s_q_vec_outside),)
            for idx, step in enumerate([4,2,0]):
                s_q_vec = s_q_vec_outside.copy()
                s_k_vec = tmp_k_vec[step:step + 2].copy()
                s_v_vec = tmp_v_vec[step:step + 2].copy()

                s_k_vec.set_color(YELLOW_E)
                s_v_vec.set_color(YELLOW_E)
                s_k_vec.set_depth(1)
                s_v_vec.set_depth(1)
                self.play(ShowCreation(s_k_vec),ShowCreation(s_v_vec),)

                self.play(
                    MoveAlongPath(s_q_vec, new_q_flash_arrow),
                    MoveAlongPath(s_k_vec, new_k_flash_arrow),
                    MoveAlongPath(s_v_vec, new_v_flash_arrow),
                )

                self.play(
                    Indicate(fused_att),
                    FadeOut(s_q_vec),
                    FadeOut(s_v_vec),
                    MoveAlongPath(s_k_vec, new_o_flash_arrow),
                )

                self.play(
                    FadeOut(s_k_vec),
                    Indicate(o),
                )
                o[step_q:step_q + 2].set_color(o_color[idx])

            s_k_vec.set_color(YELLOW_E)
            s_v_vec.set_color(YELLOW_E)
            self.play(
                FadeOut(s_q_vec_outside, run_time=0.1),
            )


        tmp_o = o.copy()
        self.play(MoveAlongPath(tmp_o, o_proj_arrow))
        self.play(Indicate(o_proj))
        self.play(MoveAlongPath(tmp_o, outputs_arrow))
        self.play(Indicate(outputs), FadeOut(tmp_o))
        outputs.set_color(GREEN)

        self.wait(2)