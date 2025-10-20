import asyncio
import platform
import pygame
import math
from math import sin, cos
from datetime import datetime
import numpy as np
import os
# Constants
cm = 8  # 4 pixels = 1 cm
mass_p1 = 0.5
mass_p2 = 0.5
v1_max = 100 * cm
v2_max = 100 * cm
f_qarsh_1_paym = 20
f_qarsh_2_paym = 20
f_glorman_1_paym = 2
f_glorman_2_paym = 2
f_sahq_paym = 10
f_aki_deform_paym_1 = 5
f_aki_deform_paym_2 = 5
# Initialize forces
f_aki_deform_1 = 0
f_aki_deform_2 = 0
f_qarsh_1 = 0
f_glorman_1 = 0
f_hrum = 0
alfa_hrum = 0
f_qarsh_2 = 0
f_glorman_2 = 0
f_aki_deform_1s = []
f_glorman_1s = []
f_qarsh_1s = []
v_1_xs = []
v_1_ys = []
klavish_1 = []
klavish_2 = []
eps = 1e-8
# --- constants & helpers ---
cm_factor = 100 * cm  # keep your original acceleration scaling
# --- helper to update each robot with the braking rule you requested ---
def update_robot_with_brake(vx, vy,
                            f_forward_vec,
                            f_glorman_paym, f_aki_deform_paym,
                            mass, v_max, dt):
    v = pygame.Vector2(vx, vy)
    # If forward force is present: opposing forces both oppose velocity as usual
    if f_forward_vec.length() > eps:
        f_glorman_vec = -v.normalize() * f_glorman_paym if v.length() > eps else pygame.Vector2(0, 0)
        f_aki_deform_vec = -v.normalize() * f_aki_deform_paym if v.length() > eps else pygame.Vector2(0,0)
        f_total = f_forward_vec + f_glorman_vec + f_aki_deform_vec
    else:
        # NO forward force: braking comes from (aki - glorman), not allowing negative braking
        brake_mag = max(f_aki_deform_paym - f_glorman_paym, 0.0)
        if v.length() > eps and brake_mag > 0.0:
            # apply braking opposite current motion
            f_brake = -v.normalize() * brake_mag
            f_total = f_brake
        else:
            # either already stopped or no net brake -> no force
            f_total = pygame.Vector2(0, 0)
    # compute acceleration using your original scaling
    a = pygame.Vector2(f_total.x / mass * cm_factor, f_total.y / mass * cm_factor)
    # semi-implicit Euler: update velocity
    v_next = v + a * dt
    # Guarantee we never reverse direction due to braking:
    # If current speed > 0 and dot(v, v_next) <= 0 then we've reached/passed zero -> snap to zero
    if v.length() > eps and v.dot(v_next) <= 0:
        v_next = pygame.Vector2(0, 0)
    # small-stop threshold to avoid jitter at extremely low speeds
    small_vel_threshold = 1e-3
    if v_next.length() < small_vel_threshold:
        v_next = pygame.Vector2(0, 0)
    # clamp magnitude WITHOUT changing direction
    speed_next = v_next.length()
    if speed_next > v_max and speed_next > eps:
        v_next.scale_to_length(v_max)
    return v_next, a, f_total

# compute the passive opposing forces magnitudes (paym values are scalars)
# note: these are used to form world-space vectors opposite the velocity when needed
def opposing_vector_from_mags(vel_vec, mag_glorman, mag_aki):
    # When forward force is absent we want braking = max(aki - glorman, 0)
    if vel_vec.length() <= eps:
        return pygame.Vector2(0, 0)
    dir_neg = -vel_vec.normalize()
    # default opposing components when forward present are both opposing vel
    return dir_neg * (mag_glorman + mag_aki)  # used when forward present

def is_point_in_rotated_square(point, center, angle, side_length=10 * cm):
    rel_x = point.x - center.x
    rel_y = point.y - center.y
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    unrotated_x = rel_x * cos_a - rel_y * sin_a
    unrotated_y = rel_x * sin_a + rel_y * cos_a
    half_side = side_length / 2
    return (-half_side <= unrotated_x <= half_side) and (-half_side <= unrotated_y <= half_side)

def calculate_dat_value(dat_pos, center, angle, step):
    max_iter = 40
    step *= cm
    point_outside = dat_pos.copy()
    length = 0
    for _ in range(max_iter):
        point_outside.x -= step * math.sin(angle)
        point_outside.y -= step * math.cos(angle)
        length += abs(step)
        if is_point_in_rotated_square(point_outside, center, angle):
            low = (length - abs(step)) / cm
            high = length / cm
            for _ in range(100):
                mid = (low + high) / 2
                test_point = dat_pos.copy()
                test_point.x -= (mid * cm) * math.sin(angle)
                test_point.y -= (mid * cm) * math.cos(angle)
                if is_point_in_rotated_square(test_point, center, angle):
                    high = mid
                else:
                    low = mid
            return (low + high) / 2
    return max_iter
def save(p1_dat1_value = None,
        p1_dat2_value = None,
        p1_dat3_value = None,
        p1_dat4_value = None,
        p1_dat5_value = None,
        p2_dat1_value = None,
        p2_dat2_value = None,
        p2_dat3_value = None,
        p2_dat4_value = None,
        p2_dat5_value = None, 
        klavish_1 = None,
        klavish_2 = None,
        winner = None):
    original_dir = os.getcwd()  # Save current directory
    save_dir = str(winner)
    os.makedirs(save_dir, exist_ok=True)  # Create winner directory if it doesn't exist
    os.chdir(save_dir)
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    p1_dat1_value = np.array(p1_dat1_value)
    p1_dat2_value = np.array(p1_dat2_value)
    p1_dat3_value = np.array(p1_dat3_value)
    p1_dat4_value = np.array(p1_dat4_value)
    p1_dat5_value = np.array(p1_dat5_value)

    p2_dat1_value = np.array(p2_dat1_value)
    p2_dat2_value = np.array(p2_dat2_value)
    p2_dat3_value = np.array(p2_dat3_value)
    p2_dat4_value = np.array(p2_dat4_value)
    p2_dat5_value = np.array(p2_dat5_value)

    klavish_1 = np.array(klavish_1)
    klavish_2 = np.array(klavish_2)
    np.savez(
        f"{winner}_{current_datetime}.npz",
        **{
            "P1_dat1": p1_dat1_value,
            "P1_dat2": p1_dat2_value,
            "P1_dat3": p1_dat3_value,
            "P1_dat4": p1_dat4_value,
            "P1_dat5": p1_dat5_value,
            "P2_dat1": p2_dat1_value,
            "P2_dat2": p2_dat2_value,
            "P2_dat3": p2_dat3_value,
            "P2_dat4": p2_dat4_value,
            "P2_dat5": p2_dat5_value,
            "Klavish_1": klavish_1,
            "Klavish_2": klavish_2,
        }
    )
    os.chdir(original_dir)
def convert(move_x, move_y):
    if move_x == 0 and move_y == 1:
        vec = [1, 1]
    elif move_x == 0 and move_y == -1:
        vec = [-1, -1]
    elif move_x == -1 and move_y == 0:
        vec = [-1, 1]
    elif move_x == 1 and move_y == 0:
        vec = [1, -1]
    else:
        vec = [0, 0]
    return vec
# Pygame setup
pygame.init()
screen = pygame.display.set_mode((700, 700))
clock = pygame.time.Clock()
FPS = 60

center = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
player1_pos = center.copy()
player2_pos = center.copy()
d_f_c = ((35 * cm) ** 2 - (15 * cm)) ** 0.5
player1_pos.y += d_f_c - 5 * cm
player2_pos.y -= d_f_c - 5 * cm
score = [0, 0]
alfa = 0
beta = 0
v_1_x = 0
v_2_x = 0
v_1_y = 0
v_2_y = 0
start_time = pygame.time.get_ticks()

p1_dat1_value = []
p1_dat2_value = []
p1_dat3_value = []
p1_dat4_value = []
p1_dat5_value = []
p2_dat1_value = []
p2_dat2_value = []
p2_dat3_value = []
p2_dat4_value = []
p2_dat5_value = []
f_lriv_1_ys = []
a_1_xs = []
a_1_ys = []
f_tot = []

async def main():
    global alfa, beta, f_qarsh_1, f_glorman_1, f_qarsh_2, f_glorman_2
    global v_1_x, v_1_y, v_2_x, v_2_y, player1_pos, player2_pos, start_time
    running = True
    while running:
        move_x_1 = 0
        move_x_2 = 0
        move_y_1 = 0
        move_y_2 = 0
        current_time = pygame.time.get_ticks()
        elapsed = current_time - start_time
        # --- physics update (replace your existing block with this) ---
        dt = clock.tick(FPS) / 1000.0  # seconds

        # convert degrees to radians once (used for force rotation)
        alfa_r = math.radians(alfa)
        beta_r = math.radians(beta)

        # --- input (turn smoothing) ---
        turn_rate = 180.0  # degrees/sec, tweak to taste
        keys = pygame.key.get_pressed()

        # player2 forward/back
        if keys[pygame.K_s] and (not keys[pygame.K_w]):
            f_qarsh_2 = -f_qarsh_2_paym
            move_y_2 -= 1
        elif keys[pygame.K_w] and (not keys[pygame.K_s]):
            f_qarsh_2 = f_qarsh_2_paym
            move_y_2 += 1
        else:
            f_qarsh_2 = 0

        # player2 rotation (smoothed)
        d_beta = 0.0
        if keys[pygame.K_a] and (not keys[pygame.K_d]) and (not keys[pygame.K_w]) and (not keys[pygame.K_s]):
            d_beta += 1.0
            move_x_2 -= 1
        if keys[pygame.K_d] and (not keys[pygame.K_a]) and (not keys[pygame.K_w]) and (not keys[pygame.K_s]):
            d_beta -= 1.0
            move_x_2 += 1
        beta += d_beta * turn_rate * dt

        # player1 forward/back
        if keys[pygame.K_k] and (not keys[pygame.K_i]):
            f_qarsh_1 = f_qarsh_1_paym
            move_y_1 -= 1
        elif keys[pygame.K_i] and (not keys[pygame.K_k]):
            f_qarsh_1 = -f_qarsh_1_paym
            move_y_1 += 1
        else:
            f_qarsh_1 = 0

        # player1 rotation (smoothed)
        d_alfa = 0.0
        if keys[pygame.K_j] and (not keys[pygame.K_l]) and (not keys[pygame.K_k]) and (not keys[pygame.K_i]):
            d_alfa += 1.0
            move_x_1 -= 1
        if keys[pygame.K_l] and (not keys[pygame.K_j]) and (not keys[pygame.K_k]) and (not keys[pygame.K_i]):
            d_alfa -= 1.0
            move_x_1 += 1
        alfa += d_alfa * turn_rate * dt

        v1 = pygame.Vector2(v_1_x, v_1_y)
        v2 = pygame.Vector2(v_2_x, v_2_y)

        # --- forward forces in local space rotated to world ---
        f_qarsh_1_local = pygame.Vector2(0, f_qarsh_1) if abs(f_qarsh_1) > 0 else pygame.Vector2(0, 0)
        f_qarsh_2_local = pygame.Vector2(0, f_qarsh_2) if abs(f_qarsh_2) > 0 else pygame.Vector2(0, 0)
        f_qarsh_1_vec = f_qarsh_1_local.rotate_rad(-alfa_r)
        f_qarsh_2_vec = f_qarsh_2_local.rotate_rad(-beta_r)

        # --- update robot 1 ---
        v1_new, a1_vec, f_total_1 = update_robot_with_brake(
            v_1_x, v_1_y, f_qarsh_1_vec,
            f_glorman_1_paym, f_aki_deform_paym_1,
            mass_p1, v1_max, dt
        )
        v_1_x, v_1_y = v1_new.x, v1_new.y

        # --- update robot 2 ---
        v2_new, a2_vec, f_total_2 = update_robot_with_brake(
            v_2_x, v_2_y, f_qarsh_2_vec,
            f_glorman_2_paym, f_aki_deform_paym_2,
            mass_p2, v2_max, dt
        )
        v_2_x, v_2_y = v2_new.x, v2_new.y

        # update positions (semi-implicit Euler)
        player1_pos.x += v_1_x * dt
        player1_pos.y += v_1_y * dt
        player2_pos.x += v_2_x * dt
        player2_pos.y += v_2_y * dt

        # --- end physics update ---
        

        # Datchikner (sensors)
        p1_dat2 = player1_pos.copy()
        p2_dat2 = player2_pos.copy()
        p2_dat2.x += 5 * cm * math.sin(beta_r)
        p2_dat2.y += 5 * cm * math.cos(beta_r)
        p1_dat2.x -= 5 * cm * math.sin(alfa_r)
        p1_dat2.y -= 5 * cm * math.cos(alfa_r)
        p1_dat1 = p1_dat2.copy()
        p1_dat3 = p1_dat2.copy()
        p2_dat1 = p2_dat2.copy()
        p2_dat3 = p2_dat2.copy()
        p1_dat1.x -= 3.5 * cm * cos(alfa_r)
        p1_dat1.y += 3.5 * cm * sin(alfa_r)
        p1_dat3.x += 3.5 * cm * cos(alfa_r)
        p1_dat3.y -= 3.5 * cm * sin(alfa_r)
        p2_dat1.x += 3.5 * cm * cos(beta_r)
        p2_dat1.y -= 3.5 * cm * sin(beta_r)
        p2_dat3.x -= 3.5 * cm * cos(beta_r)
        p2_dat3.y += 3.5 * cm * sin(beta_r)
        p1_dat4 = player1_pos.copy()
        p1_dat4.x += 5 * cm * cos(alfa_r) - 3 * sin(alfa_r) * cm
        p1_dat4.y -= 5 * cm * sin(alfa_r) + 3 * cos(alfa_r) * cm
        p1_dat5 = p1_dat4.copy()
        p1_dat5.x -= 10 * cm * cos(alfa_r)
        p1_dat5.y += 10 * cm * sin(alfa_r)
        p2_dat4 = player2_pos.copy()
        p2_dat4.x -= 5 * cm * cos(beta_r) - 3 * sin(beta_r) * cm
        p2_dat4.y += 5 * cm * sin(beta_r) + 3 * cos(beta_r) * cm
        p2_dat5 = p2_dat4.copy()
        p2_dat5.x += 10 * cm * cos(beta_r)
        p2_dat5.y -= 10 * cm * sin(beta_r)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Drawing
        screen.fill("white")
        pygame.draw.circle(screen, "yellow", center, 50 * cm)
        pygame.draw.circle(screen, "white", center, 36 * cm)
        pygame.draw.circle(screen, "black", center, 35 * cm)
        pygame.draw.circle(screen, "white", p1_dat1, 2)
        pygame.draw.circle(screen, "white", p1_dat2, 2)
        pygame.draw.circle(screen, "white", p1_dat3, 2)
        pygame.draw.circle(screen, "white", p1_dat4, 2)
        pygame.draw.circle(screen, "white", p1_dat5, 2)
        pygame.draw.circle(screen, "white", p2_dat1, 2)
        pygame.draw.circle(screen, "white", p2_dat2, 2)
        pygame.draw.circle(screen, "white", p2_dat3, 2)
        pygame.draw.circle(screen, "white", p2_dat4, 2)
        pygame.draw.circle(screen, "white", p2_dat5, 2)
        p1_surf = pygame.Surface((10 * cm, 10 * cm), pygame.SRCALPHA)
        pygame.draw.rect(p1_surf, "green", (0, 0, 10 * cm, 10 * cm))
        p1_rotated = pygame.transform.rotate(p1_surf, alfa)
        p1_rect = p1_rotated.get_rect(center=(player1_pos.x, player1_pos.y))
        screen.blit(p1_rotated, p1_rect)
        p2_surf = pygame.Surface((10 * cm, 10 * cm), pygame.SRCALPHA)
        pygame.draw.rect(p2_surf, "red", (0, 0, 10 * cm, 10 * cm))
        p2_rotated = pygame.transform.rotate(p2_surf, beta)
        p2_rect = p2_rotated.get_rect(center=(player2_pos.x, player2_pos.y))
        screen.blit(p2_rotated, p2_rect)

        # Collision detection with bounce
        # --- Collision detection & impulse resolution (replace your simple bounce block) ---
        p1_mask = pygame.mask.from_surface(p1_rotated)
        p2_mask = pygame.mask.from_surface(p2_rotated)
        offset = (p2_rect.left - p1_rect.left, p2_rect.top - p1_rect.top)
        overlap_point = p1_mask.overlap(p2_mask, offset)  # returns local coords in p1_mask or None

        if overlap_point:
            # Tuning params
            restitution = 0.3       # bounciness: 0.0 = inelastic, 1.0 = fully elastic (tweak)
            positional_percent = 0.2  # percent of penetration to correct (0.2 is typical)
            slop = 0.5               # penetration allowance in pixels (avoid jitter)
            eps = 1e-8

            # Compute collision normal: prefer center-to-center as stable approximation
            center_vec = pygame.Vector2(player2_pos.x - player1_pos.x, player2_pos.y - player1_pos.y)
            dist_centers = center_vec.length()
            if dist_centers > eps:
                normal = center_vec.normalize()
            else:
                # fallback: try using difference of rotations (rare)
                ang_diff = math.radians(beta - alfa)
                normal = pygame.Vector2(math.sin(ang_diff), math.cos(ang_diff))
                if normal.length() < eps:
                    normal = pygame.Vector2(0, -1)  # ultimate fallback

            # Relative velocity at centers (you don't model angular velocity)
            v1 = pygame.Vector2(v_1_x, v_1_y)
            v2 = pygame.Vector2(v_2_x, v_2_y)
            rel_vel = v2 - v1

            # Velocity along normal
            vel_along_normal = rel_vel.dot(normal)

            # If velocities are separating, do nothing
            if vel_along_normal < 0:
                inv_m1 = 1.0 / mass_p1 if mass_p1 > eps else 0.0
                inv_m2 = 1.0 / mass_p2 if mass_p2 > eps else 0.0

                # impulse scalar (1D collision along normal)
                j = -(1 + restitution) * vel_along_normal
                denom = inv_m1 + inv_m2
                if denom > eps:
                    j /= denom

                    impulse = normal * j

                    # apply impulses (change linear velocities)
                    v_1_x -= (impulse.x * inv_m1)
                    v_1_y -= (impulse.y * inv_m1)
                    v_2_x += (impulse.x * inv_m2)
                    v_2_y += (impulse.y * inv_m2)

            # Positional correction to avoid sinking/overlap:
            # approximate penetration using square-to-circle conservative radii (robots are 10*cm squares)
            side = 10 * cm
            # radius of circumscribed circle for square: half-diagonal = side / sqrt(2)
            r = side / (2 ** 0.5)
            sum_r = r + r
            # distance between centers (recompute robustly)
            dist = player1_pos.distance_to(player2_pos)
            penetration = max(0.0, sum_r - dist)

            if penetration > slop and (inv_m1 + inv_m2) > eps:
                correction_magnitude = (max(penetration - slop, 0.0) / (inv_m1 + inv_m2)) * positional_percent
                correction = normal * correction_magnitude
                # move objects apart proportionally to their inverse masses
                player1_pos -= correction * inv_m1
                player2_pos += correction * inv_m2

            # Optional: reduce velocities a bit to simulate energy loss / friction on contact
            contact_damping = 0.98
            v_1_x *= contact_damping
            v_1_y *= contact_damping
            v_2_x *= contact_damping
            v_2_y *= contact_damping

            # (Optional) You can show the contact point for debugging:
            # pygame.draw.circle(screen, "blue", contact_world, 3)

        # Scoring
        dist_p1 = player1_pos.distance_to(center)
        dist_p2 = player2_pos.distance_to(center)
        if dist_p1 > 36 * cm:
            score[0] += 1
            player1_pos = center.copy()
            player2_pos = center.copy()
            player1_pos.y += d_f_c - 5 * cm
            player2_pos.y -= d_f_c - 5 * cm
            alfa = 0
            beta = 0
            v_1_x = v_1_y = v_2_x = v_2_y = 0  # Reset velocities
            save(
                p1_dat1_value,
                p1_dat2_value,
                p1_dat3_value,
                p1_dat4_value,
                p1_dat5_value,
                p2_dat1_value,
                p2_dat2_value,
                p2_dat3_value,
                p2_dat4_value,
                p2_dat5_value,
                klavish_1,
                klavish_2,
                2
            )
            print(f"Score {score[0]} : {score[1]}")
        if dist_p2 > 36 * cm:
            score[1] += 1
            player1_pos = center.copy()
            player2_pos = center.copy()
            player1_pos.y += d_f_c - 5 * cm
            player2_pos.y -= d_f_c - 5 * cm
            alfa = 0
            beta = 0
            v_1_x = v_1_y = v_2_x = v_2_y = 0
            save(
                p1_dat1_value,
                p1_dat2_value,
                p1_dat3_value,
                p1_dat4_value,
                p1_dat5_value,
                p2_dat1_value,
                p2_dat2_value,
                p2_dat3_value,
                p2_dat4_value,
                p2_dat5_value,
                klavish_1,
                klavish_2,
                1
            )  # Reset velocities
            print(f"Score {score[0]} : {score[1]}")
        # Sensor data collection
        if elapsed > 100:
            klavish_1.append(convert(move_x_1,move_y_1))
            klavish_2.append(convert(move_x_2,move_y_2))
            val = calculate_dat_value(p1_dat1, player2_pos, alfa_r, 1)
            p1_dat1_value.append(val)
            val = calculate_dat_value(p1_dat2, player2_pos, alfa_r, 1)
            p1_dat2_value.append(val)
            val = calculate_dat_value(p1_dat3, player2_pos, alfa_r, 1)
            p1_dat3_value.append(val)
            val = calculate_dat_value(p1_dat4, player2_pos, alfa_r, 1)
            p1_dat4_value.append(val)
            val = calculate_dat_value(p1_dat5, player2_pos, alfa_r, 1)
            p1_dat5_value.append(val)
            val = calculate_dat_value(p2_dat1, player1_pos, beta_r, -1)
            p2_dat1_value.append(abs(val))
            val = calculate_dat_value(p2_dat2, player1_pos, beta_r, -1)
            p2_dat2_value.append(abs(val))
            val = calculate_dat_value(p2_dat3, player1_pos, beta_r, -1)
            p2_dat3_value.append(abs(val))
            val = calculate_dat_value(p2_dat4, player1_pos, beta_r, -1)
            p2_dat4_value.append(abs(val))
            val = calculate_dat_value(p2_dat5, player1_pos, beta_r, -1)
            p2_dat5_value.append(abs(val))
            start_time = current_time
        pygame.display.flip()
        await asyncio.sleep(1.0 / FPS)

    #print(f"Score {score[0]} : {score[1]}")
    #print(f"P1 dat1 : {p1_dat1_value}")
    #print(f"P1 dat2 : {p1_dat2_value}")
    #print(f"P1 dat3 : {p1_dat3_value}")
    #print(f"P1 dat4 : {p1_dat4_value}")
    #print(f"P1 dat5 : {p1_dat5_value}")
    #print(f"P2 dat1 : {p2_dat1_value}")
    #print(f"P2 dat2 : {p2_dat2_value}")
    #print(f"P2 dat3 : {p2_dat3_value}")
    #print(f"P2 dat4 : {p2_dat4_value}")
    #print(f"P2 dat5 : {p2_dat5_value}")
    #print(f"Klavish_1 : {klavish_1}")
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())