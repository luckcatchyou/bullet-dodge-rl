import random
from collections import deque

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, MultiDiscrete

from .utils import clamp, dist_circle_to_rect, norm

FPS = 60
DT = 1 / FPS
WORLD_W, WORLD_H = 2400, 2400
SCREEN_W, SCREEN_H = 1920, 1080

CELL = 40
GRID_W = SCREEN_W // CELL
GRID_H = SCREEN_H // CELL
SAFE_WALL_DIST = 200

PLAYER_HP = 10
PLAYER_SPEED = 400
PLAYER_W = 50
PLAYER_H = 70
HIT_INVINCIBLE_DURATION = 0

DASH_SPEED_MULT = 1.2
DASH_DURATION = 0.6
DASH_COOLDOWN = 1
DASH_RECOVERY_DURATION = 0.1
DASH_RECOVERY_SPEED = 0.5

BULLET_R = 20
BULLET_SPEED = 550
BULLET_SPAWN_RATE = 2
BULLET_RANDOMNESS = 0.05

ENEMY_COUNT = 4
ENEMY_W = 60
ENEMY_H = 80
ENEMY_SPEED = 120
ENEMY_SHOOT_CD = 1.2
ENEMY_SHOOT_SPREAD = 0.15
ENEMY_DIR_CHANGE_PROB = 0.01


class Game:
    def __init__(self, render_enabled=True, debug_obs=False, player_max_hp=PLAYER_HP):
        self.debug_obs = debug_obs
        self.render_enabled = render_enabled
        if render_enabled:
            pygame.init()
            pygame.key.stop_text_input()
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            self.clock = pygame.time.Clock()

        self.player_max_hp = player_max_hp
        self.bullets = []
        self.enemies = []
        self.steps = 0
        self.max_steps = 0
        self.reset(clear_bullets=True)
        self._reset_enemies()

    def _reset_enemies(self):
        self.enemies = []
        for _ in range(ENEMY_COUNT):
            vx, vy = norm(random.uniform(-1, 1), random.uniform(-1, 1))
            self.enemies.append(
                {
                    "x": random.uniform(300, WORLD_W - 300),
                    "y": random.uniform(300, WORLD_H - 300),
                    "vx": vx,
                    "vy": vy,
                    "cooldown": random.uniform(0, ENEMY_SHOOT_CD),
                }
            )

    def reset(self, clear_bullets=False):
        self.max_steps = max(self.max_steps, self.steps)
        self.steps = 0
        self.player_hp = self.player_max_hp
        self.hit_invincible_timer = 0
        self.is_invincible = False
        self.player_x = WORLD_W / 2
        self.player_y = WORLD_H / 2
        self.dash_timer = 0
        self.dash_cooldown_timer = 0
        self.dash_vx = 0
        self.dash_vy = 0

        if clear_bullets:
            self.bullets = []

    def _spawn_random_bullet(self):
        side = random.choice(["l", "r", "t", "b"])
        if side == "l":
            x, y = 0, random.uniform(0, WORLD_H)
        elif side == "r":
            x, y = WORLD_W, random.uniform(0, WORLD_H)
        elif side == "t":
            x, y = random.uniform(0, WORLD_W), 0
        else:
            x, y = random.uniform(0, WORLD_W), WORLD_H

        tx, ty = norm(self.player_x - x, self.player_y - y)
        rx, ry = norm(random.uniform(-1, 1), random.uniform(-1, 1))
        vx, vy = norm(tx * (1 - BULLET_RANDOMNESS) + rx * BULLET_RANDOMNESS,
                      ty * (1 - BULLET_RANDOMNESS) + ry * BULLET_RANDOMNESS)
        return {"x": x, "y": y, "vx": vx * BULLET_SPEED, "vy": vy * BULLET_SPEED}

    def _spawn_enemy_bullet(self, enemy):
        tx, ty = norm(self.player_x - enemy["x"], self.player_y - enemy["y"])
        rx, ry = norm(random.uniform(-1, 1), random.uniform(-1, 1))
        vx, vy = norm(tx * (1 - ENEMY_SHOOT_SPREAD) + rx * ENEMY_SHOOT_SPREAD,
                      ty * (1 - ENEMY_SHOOT_SPREAD) + ry * ENEMY_SHOOT_SPREAD)
        return {"x": enemy["x"], "y": enemy["y"], "vx": vx * BULLET_SPEED, "vy": vy * BULLET_SPEED}

    def step(self, action, dash_requested=False):
        self.steps += 1
        dx = dy = 0
        if action == 1:
            dy -= 1
        elif action == 2:
            dy += 1
        elif action == 3:
            dx -= 1
        elif action == 4:
            dx += 1
        elif action == 5:
            dx -= 1
            dy -= 1
        elif action == 6:
            dx += 1
            dy -= 1
        elif action == 7:
            dx -= 1
            dy += 1
        elif action == 8:
            dx += 1
            dy += 1
        dx, dy = norm(dx, dy)

        self.hit_invincible_timer = max(self.hit_invincible_timer - DT, 0)
        self.dash_cooldown_timer = max(self.dash_cooldown_timer - DT, 0)
        self.dash_timer = max(self.dash_timer - DT, 0)

        if dash_requested and self.dash_timer <= 0 and self.dash_cooldown_timer <= 0 and (dx != 0 or dy != 0):
            self.dash_timer = DASH_DURATION
            self.dash_cooldown_timer = DASH_COOLDOWN
            self.dash_vx, self.dash_vy = dx, dy

        self.is_invincible = self.hit_invincible_timer > 0
        if self.dash_timer > DASH_RECOVERY_DURATION:
            current_vx = self.dash_vx * PLAYER_SPEED * DASH_SPEED_MULT
            current_vy = self.dash_vy * PLAYER_SPEED * DASH_SPEED_MULT
            self.is_invincible = True
        elif self.dash_timer > 0:
            current_vx = self.dash_vx * PLAYER_SPEED * DASH_RECOVERY_SPEED
            current_vy = self.dash_vy * PLAYER_SPEED * DASH_RECOVERY_SPEED
        else:
            current_vx = dx * PLAYER_SPEED
            current_vy = dy * PLAYER_SPEED

        self.player_x = clamp(self.player_x + current_vx * DT, PLAYER_W / 2, WORLD_W - PLAYER_W / 2)
        self.player_y = clamp(self.player_y + current_vy * DT, PLAYER_H / 2, WORLD_H - PLAYER_H / 2)

        if random.random() < BULLET_SPAWN_RATE * DT:
            self.bullets.append(self._spawn_random_bullet())

        is_hit = False
        for enemy in self.enemies:
            if random.random() < ENEMY_DIR_CHANGE_PROB:
                enemy["vx"], enemy["vy"] = norm(random.uniform(-1, 1), random.uniform(-1, 1))

            enemy["x"] += enemy["vx"] * ENEMY_SPEED * DT
            enemy["y"] += enemy["vy"] * ENEMY_SPEED * DT

            if enemy["x"] < ENEMY_W / 2 or enemy["x"] > WORLD_W - ENEMY_W / 2:
                enemy["vx"] *= -1
            if enemy["y"] < ENEMY_H / 2 or enemy["y"] > WORLD_H - ENEMY_H / 2:
                enemy["vy"] *= -1

            enemy["x"] = clamp(enemy["x"], ENEMY_W / 2, WORLD_W - ENEMY_W / 2)
            enemy["y"] = clamp(enemy["y"], ENEMY_H / 2, WORLD_H - ENEMY_H / 2)

            if not self.is_invincible and self._player_overlaps_enemy(enemy):
                is_hit = True

            enemy["cooldown"] -= DT
            if enemy["cooldown"] <= 0:
                self.bullets.append(self._spawn_enemy_bullet(enemy))
                enemy["cooldown"] = ENEMY_SHOOT_CD

        self._update_bullets(is_hit)

        if self.render_enabled:
            self.render(is_hit)

    def _player_overlaps_enemy(self, enemy):
        p_left = self.player_x - PLAYER_W / 2
        p_right = self.player_x + PLAYER_W / 2
        p_top = self.player_y - PLAYER_H / 2
        p_bottom = self.player_y + PLAYER_H / 2
        e_left = enemy["x"] - ENEMY_W / 2
        e_right = enemy["x"] + ENEMY_W / 2
        e_top = enemy["y"] - ENEMY_H / 2
        e_bottom = enemy["y"] + ENEMY_H / 2
        return p_left < e_right and p_right > e_left and p_top < e_bottom and p_bottom > e_top

    def _update_bullets(self, already_hit):
        new_bullets = []
        is_hit = already_hit
        for bullet in self.bullets:
            bullet["x"] += bullet["vx"] * DT
            bullet["y"] += bullet["vy"] * DT
            dist = dist_circle_to_rect(bullet["x"], bullet["y"], self.player_x, self.player_y, PLAYER_W, PLAYER_H)

            if dist < BULLET_R and not self.is_invincible:
                is_hit = True
                continue

            if 0 <= bullet["x"] <= WORLD_W and 0 <= bullet["y"] <= WORLD_H:
                new_bullets.append(bullet)

        self.bullets = new_bullets
        if self.player_hp != -1 and is_hit:
            self.player_hp -= 1
            self.hit_invincible_timer = HIT_INVINCIBLE_DURATION

    def get_obs_single_frame(self):
        grid = np.zeros((GRID_H, GRID_W), np.float32)
        p_x, p_y = self.player_x, self.player_y
        half_w, half_h = GRID_W // 2, GRID_H // 2
        cam_x = p_x - SCREEN_W / 2
        cam_y = p_y - SCREEN_H / 2

        for gy in range(GRID_H):
            for gx in range(GRID_W):
                wx = p_x + (gx - half_w) * CELL
                wy = p_y + (gy - half_h) * CELL
                dist = min(wx, WORLD_W - wx, wy, WORLD_H - wy)
                if dist < SAFE_WALL_DIST:
                    grid[gy, gx] = max(grid[gy, gx], 1.0)

        for bullet in self.bullets:
            sx = bullet["x"] - cam_x
            sy = bullet["y"] - cam_y
            if sx < -BULLET_R or sx > SCREEN_W + BULLET_R or sy < -BULLET_R or sy > SCREEN_H + BULLET_R:
                continue

            gx0 = int((sx - BULLET_R) // CELL)
            gx1 = int((sx + BULLET_R) // CELL)
            gy0 = int((sy - BULLET_R) // CELL)
            gy1 = int((sy + BULLET_R) // CELL)
            for gy in range(max(0, gy0), min(GRID_H, gy1 + 1)):
                for gx in range(max(0, gx0), min(GRID_W, gx1 + 1)):
                    grid[gy, gx] = max(grid[gy, gx], 2.0)

        for enemy in self.enemies:
            left = enemy["x"] - ENEMY_W / 2 - cam_x
            right = enemy["x"] + ENEMY_W / 2 - cam_x
            top = enemy["y"] - ENEMY_H / 2 - cam_y
            bottom = enemy["y"] + ENEMY_H / 2 - cam_y
            gx0 = int(left // CELL)
            gx1 = int(right // CELL)
            gy0 = int(top // CELL)
            gy1 = int(bottom // CELL)
            for gy in range(gy0, gy1 + 1):
                for gx in range(gx0, gx1 + 1):
                    if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                        grid[gy, gx] = max(grid[gy, gx], 3.0)

        return grid / 4

    def render(self, is_hit=False):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.clock.tick(FPS)
        self.screen.fill((40, 40, 40))
        cam_x = self.player_x - SCREEN_W / 2
        cam_y = self.player_y - SCREEN_H / 2

        for x in range(0, WORLD_W, 80):
            pygame.draw.line(self.screen, (60, 60, 60), (x - cam_x, -cam_y), (x - cam_x, WORLD_H - cam_y))
        for y in range(0, WORLD_H, 80):
            pygame.draw.line(self.screen, (60, 60, 60), (-cam_x, y - cam_y), (WORLD_W - cam_x, y - cam_y))

        pygame.draw.rect(self.screen, (200, 200, 200), (-cam_x, -cam_y, WORLD_W, WORLD_H), 4)

        for bullet in self.bullets:
            pygame.draw.circle(self.screen, (255, 80, 80), (int(bullet["x"] - cam_x), int(bullet["y"] - cam_y)), BULLET_R)

        for enemy in self.enemies:
            rect = pygame.Rect(int(enemy["x"] - ENEMY_W / 2 - cam_x), int(enemy["y"] - ENEMY_H / 2 - cam_y), ENEMY_W, ENEMY_H)
            pygame.draw.rect(self.screen, (220, 140, 80), rect)

        player_rect = pygame.Rect(SCREEN_W // 2 - PLAYER_W // 2, SCREEN_H // 2 - PLAYER_H // 2, PLAYER_W, PLAYER_H)
        color = (80, 200, 255)
        if is_hit:
            color = (255, 0, 0)
        elif self.dash_timer > 0:
            color = (255, 255, 255)
        pygame.draw.rect(self.screen, color, player_rect)

        font = pygame.font.SysFont("Arial", 24)
        hp_text = "inf" if self.player_hp == -1 else str(self.player_hp)
        self.screen.blit(font.render(f"HP: {hp_text}", True, (0, 255, 0)), (100, 30))
        self.screen.blit(font.render(f"Steps: {self.steps}", True, (255, 255, 255)), (100, 60))
        self.screen.blit(font.render(f"Max Steps: {self.max_steps}", True, (255, 255, 255)), (100, 90))

        if self.debug_obs:
            self._render_debug_grid()

        self._render_controls()

        pygame.display.flip()

    def _render_controls(self):
        key_size = 36
        gap = 6
        origin_x = SCREEN_W - 150
        origin_y = 30

        positions = {
            "W": (origin_x + key_size + gap, origin_y),
            "A": (origin_x, origin_y + key_size + gap),
            "S": (origin_x + key_size + gap, origin_y + key_size + gap),
            "D": (origin_x + (key_size + gap) * 2, origin_y + key_size + gap),
        }

        font = pygame.font.SysFont("Arial", 18, bold=True)

        for label, (lx, ly) in positions.items():
            rect = pygame.Rect(lx, ly, key_size, key_size)
            shadow = rect.copy()
            shadow.y += 3
            pygame.draw.rect(self.screen, (20, 20, 20), shadow, border_radius=6)
            pygame.draw.rect(self.screen, (70, 70, 70), rect, border_radius=6)
            pygame.draw.rect(self.screen, (130, 130, 130), rect, width=1, border_radius=6)

            text = font.render(label, True, (220, 220, 220))
            text_rect = text.get_rect(center=(lx + key_size // 2, ly + key_size // 2))
            self.screen.blit(text, text_rect)

        space_w = key_size * 3 + gap * 2
        space_h = 28
        space_x = origin_x
        space_y = origin_y + (key_size + gap) * 2 + 12
        space_rect = pygame.Rect(space_x, space_y, space_w, space_h)
        shadow = space_rect.copy()
        shadow.y += 3
        pygame.draw.rect(self.screen, (20, 20, 20), shadow, border_radius=8)
        pygame.draw.rect(self.screen, (70, 70, 70), space_rect, border_radius=8)
        pygame.draw.rect(self.screen, (130, 130, 130), space_rect, width=1, border_radius=8)

        space_label = font.render("Space", True, (220, 220, 220))
        space_label_rect = space_label.get_rect(center=(space_x + space_w // 2, space_y + space_h // 2))
        self.screen.blit(space_label, space_label_rect)

    def _render_debug_grid(self):
        debug_grid = self.get_obs_single_frame() * 4
        grid_surf = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        colors = {
            1: (255, 255, 0, 80),
            2: (255, 0, 0, 120),
            3: (255, 0, 255, 100),
        }
        for gy in range(GRID_H):
            for gx in range(GRID_W):
                val = round(debug_grid[gy, gx])
                if val > 0:
                    color = colors.get(val, (255, 255, 255, 50))
                    rect = (gx * CELL, gy * CELL, CELL, CELL)
                    pygame.draw.rect(grid_surf, color, rect)
                    pygame.draw.rect(grid_surf, (color[0], color[1], color[2], 150), rect, 1)
        self.screen.blit(grid_surf, (0, 0))

    def play(self):
        while True:
            dash_requested = False
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    dash_requested = True
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    return

            keys = pygame.key.get_pressed()
            action = 0
            if keys[pygame.K_w] and keys[pygame.K_a]:
                action = 5
            elif keys[pygame.K_w] and keys[pygame.K_d]:
                action = 6
            elif keys[pygame.K_s] and keys[pygame.K_a]:
                action = 7
            elif keys[pygame.K_s] and keys[pygame.K_d]:
                action = 8
            elif keys[pygame.K_w]:
                action = 1
            elif keys[pygame.K_s]:
                action = 2
            elif keys[pygame.K_a]:
                action = 3
            elif keys[pygame.K_d]:
                action = 4

            self.step(action, dash_requested)
            if self.player_hp <= 0:
                self.reset()


class BulletEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_enabled=False, debug_obs=False, max_episode_steps=float("inf"), stack_size=4):
        super().__init__()
        self.game = Game(render_enabled=render_enabled, debug_obs=debug_obs)
        self.action_space = MultiDiscrete([9, 2])
        self.observation_space = Box(0, 4, shape=(stack_size, GRID_H, GRID_W), dtype=np.float32)
        self.stack_size = stack_size
        self.obs_queue = deque(maxlen=stack_size)
        self.max_episode_steps = max_episode_steps
        self.last_health = PLAYER_HP

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_health = PLAYER_HP
        self.game.reset(clear_bullets=True)
        self.game._reset_enemies()

        grid = np.zeros((GRID_H, GRID_W), np.float32)
        for _ in range(self.stack_size):
            self.obs_queue.append(grid)

        return self._get_obs(), {}

    def _get_obs(self):
        self.obs_queue.append(self.game.get_obs_single_frame())
        return np.array(self.obs_queue, dtype=np.float32)

    def step(self, action):
        for _ in range(6):
            self.game.step(int(action[0]), bool(action[1]))

        truncated = self.game.steps >= self.max_episode_steps
        reward, terminated = self.compute_reward()
        if action[1]:
            reward -= 0.5

        return self._get_obs(), reward, terminated, truncated, {"steps": self.game.steps}

    def compute_reward(self):
        reward = 0.1

        if self.game.bullets:
            min_dist = min(
                dist_circle_to_rect(b["x"], b["y"], self.game.player_x, self.game.player_y, PLAYER_W, PLAYER_H)
                for b in self.game.bullets
            )
            min_radius = BULLET_R
            max_radius = 5 * min_radius
            min_reward = -1.0
            max_reward = 0.1
            if min_dist <= min_radius:
                reward += min_reward
            elif min_dist >= max_radius:
                reward += max_reward
            else:
                t = (min_dist - min_radius) / (max_radius - min_radius)
                reward += min_reward * (1 - t) ** 2 + max_reward * (t ** 2)

        dist_to_wall = min(
            self.game.player_x,
            WORLD_W - self.game.player_x,
            self.game.player_y,
            WORLD_H - self.game.player_y,
        )
        if dist_to_wall < SAFE_WALL_DIST:
            t = dist_to_wall / SAFE_WALL_DIST
            reward -= (1 - t) ** 2

        terminated = False
        current_health = self.game.player_hp
        if current_health < self.last_health:
            reward -= 5
        if current_health <= 0:
            terminated = True
        self.last_health = current_health

        return reward, terminated

    def close(self):
        if self.game.render_enabled:
            pygame.quit()
