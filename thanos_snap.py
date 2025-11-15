import cv2
import numpy as np
import mediapipe as mp
import time
import math
import random

class EnhancedThanosSnap:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.hand_history = []
        self.snap_threshold = 0.7
        self.last_snap_time = 0
        self.snap_cooldown = 2
        
        self.background = None
        self.background_captured = False
        self.is_invisible = False
        self.invisibility_start_time = 0
        self.invisibility_duration = 20
        
        self.dust_particles = []
        self.energy_particles = []
        self.impact_rings = []
        self.snap_effect_active = False
        
        self.dust_colors = [(139, 69, 19), (160, 82, 45), (210, 180, 140), (222, 184, 135), (205, 133, 63)]
        self.energy_colors = [(0, 255, 255), (255, 255, 0), (255, 165, 0), (255, 215, 0)]
    
    def detect_snap_gesture(self, landmarks, frame_width, frame_height):
        if landmarks is None:
            return False, 0, None
            
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        thumb_pos = [thumb_tip.x * frame_width, thumb_tip.y * frame_height]
        index_pos = [index_tip.x * frame_width, index_tip.y * frame_height]
        
        distance = math.sqrt((thumb_pos[0] - index_pos[0])**2 + (thumb_pos[1] - index_pos[1])**2)
        
        current_time = time.time()
        if current_time - self.last_snap_time < self.snap_cooldown:
            return False, distance / 100, thumb_pos
        
        if distance < 25:
            confidence = 1.0 - (distance / 25)
            if confidence >= self.snap_threshold:
                self.last_snap_time = current_time
                return True, confidence, thumb_pos
        
        return False, distance / 100, thumb_pos
    
    def capture_background(self):
        for i in range(5, 0, -1):
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"MOVE OUT OF FRAME: {i}", (300, 350), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.imshow('Background Capture', frame)
                cv2.waitKey(1)
            time.sleep(1)
        
        background_frames = []
        for i in range(50):
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                background_frames.append(frame.copy())
                
                progress = int((i + 1) / 50 * 100)
                cv2.putText(frame, f"Capturing: {progress}%", (400, 350), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow('Background Capture', frame)
                cv2.waitKey(30)
        
        if background_frames:
            self.background = np.median(background_frames, axis=0).astype(np.uint8)
            self.background = cv2.GaussianBlur(self.background, (3, 3), 0)
            self.background_captured = True
        
        cv2.destroyWindow('Background Capture')
    
    def create_person_mask(self, current_frame):
        if self.background is None:
            return np.zeros(current_frame.shape[:2], dtype=np.uint8)
        
        diff = cv2.absdiff(current_frame, self.background)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        _, mask1 = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)
        _, mask2 = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
        _, mask3 = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
        
        combined_mask = cv2.bitwise_or(mask1, mask2)
        combined_mask = cv2.bitwise_or(combined_mask, mask3)
        
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_medium)
        combined_mask = cv2.dilate(combined_mask, kernel_large, iterations=6)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            clean_mask = np.zeros(combined_mask.shape, dtype=np.uint8)
            cv2.fillPoly(clean_mask, [largest_contour], 255)
            
            clean_mask = cv2.GaussianBlur(clean_mask, (7, 7), 0)
            _, clean_mask = cv2.threshold(clean_mask, 127, 255, cv2.THRESH_BINARY)
            
            return clean_mask
        
        return combined_mask
    
    def apply_invisibility(self, current_frame):
        if not self.background_captured:
            return current_frame
        
        person_mask = self.create_person_mask(current_frame)
        person_mask_smooth = cv2.GaussianBlur(person_mask, (9, 9), 0)
        
        mask_3channel = cv2.cvtColor(person_mask_smooth, cv2.COLOR_GRAY2BGR) / 255.0
        inverse_mask_3channel = 1.0 - mask_3channel
        
        blended = (self.background.astype(float) * mask_3channel + 
                  current_frame.astype(float) * inverse_mask_3channel)
        
        return blended.astype(np.uint8)
    
    def create_dust_explosion(self, center, frame_shape):
        for cluster in range(15):
            angle = (cluster / 15) * 2 * math.pi
            for _ in range(20):
                x = center[0] + math.cos(angle) * random.uniform(20, 80)
                y = center[1] + math.sin(angle) * random.uniform(20, 80)
                
                particle = {
                    'x': x, 'y': y,
                    'vel_x': math.cos(angle) * random.uniform(2, 6),
                    'vel_y': math.sin(angle) * random.uniform(2, 6),
                    'life': random.randint(60, 120),
                    'max_life': random.randint(60, 120),
                    'size': random.uniform(3, 8),
                    'color': random.choice(self.dust_colors),
                    'rotation': random.uniform(0, 2 * math.pi),
                    'rotation_speed': random.uniform(-0.1, 0.1)
                }
                self.dust_particles.append(particle)
    
    def create_energy_wave(self, center):
        for _ in range(40):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(4, 10)
            
            particle = {
                'x': center[0], 'y': center[1],
                'vel_x': math.cos(angle) * speed,
                'vel_y': math.sin(angle) * speed,
                'life': random.randint(40, 80),
                'max_life': random.randint(40, 80),
                'size': random.uniform(2, 6),
                'color': random.choice(self.energy_colors)
            }
            self.energy_particles.append(particle)
        
        for i in range(5):
            ring = {
                'x': center[0], 'y': center[1],
                'radius': 15 + i * 8,
                'max_radius': random.randint(200, 400),
                'expand_speed': random.uniform(4, 12),
                'life': random.randint(30, 70),
                'max_life': random.randint(30, 70),
                'color': random.choice(self.energy_colors),
                'thickness': random.randint(3, 7)
            }
            self.impact_rings.append(ring)
    
    def update_effects(self, frame):
        for i in range(len(self.dust_particles) - 1, -1, -1):
            p = self.dust_particles[i]
            p['x'] += p['vel_x']
            p['y'] += p['vel_y']
            p['vel_y'] += 0.15
            p['vel_x'] *= 0.99
            p['vel_y'] *= 0.99
            p['rotation'] += p['rotation_speed']
            p['life'] -= 1
            
            if p['life'] <= 0 or p['y'] > frame.shape[0] + 50:
                self.dust_particles.pop(i)
            else:
                alpha = p['life'] / p['max_life']
                size = int(p['size'] * alpha)
                
                if size > 0:
                    x, y = int(p['x']), int(p['y'])
                    cos_rot = math.cos(p['rotation'])
                    sin_rot = math.sin(p['rotation'])
                    
                    half_size = size // 2
                    corners = [(-half_size, -half_size), (half_size, -half_size), 
                              (half_size, half_size), (-half_size, half_size)]
                    
                    rotated_corners = []
                    for cx, cy in corners:
                        rx = cx * cos_rot - cy * sin_rot + x
                        ry = cx * sin_rot + cy * cos_rot + y
                        rotated_corners.append([int(rx), int(ry)])
                    
                    pts = np.array(rotated_corners, np.int32)
                    cv2.fillPoly(frame, [pts], p['color'])
        
        for i in range(len(self.energy_particles) - 1, -1, -1):
            p = self.energy_particles[i]
            p['x'] += p['vel_x']
            p['y'] += p['vel_y']
            p['life'] -= 1
            
            if p['life'] <= 0:
                self.energy_particles.pop(i)
            else:
                alpha = p['life'] / p['max_life']
                size = int(p['size'] * alpha)
                
                if size > 0:
                    x, y = int(p['x']), int(p['y'])
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        color = tuple(int(c * alpha) for c in p['color'])
                        cv2.circle(frame, (x, y), size + 2, color, -1)
                        cv2.circle(frame, (x, y), max(1, size), (255, 255, 255), -1)
        
        for i in range(len(self.impact_rings) - 1, -1, -1):
            ring = self.impact_rings[i]
            ring['radius'] += ring['expand_speed']
            ring['life'] -= 1
            
            if ring['life'] <= 0 or ring['radius'] > ring['max_radius']:
                self.impact_rings.pop(i)
            else:
                alpha = ring['life'] / ring['max_life']
                thickness = max(1, int(ring['thickness'] * alpha))
                color = tuple(int(c * alpha) for c in ring['color'])
                cv2.circle(frame, (int(ring['x']), int(ring['y'])), 
                          int(ring['radius']), color, thickness)
        
        if len(self.dust_particles) == 0 and len(self.energy_particles) == 0 and len(self.impact_rings) == 0:
            self.snap_effect_active = False
    
    def trigger_snap_effects(self, position):
        self.snap_effect_active = True
        self.create_dust_explosion(position, (720, 1280))
        self.create_energy_wave(position)
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks and self.background_captured:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    is_snap, confidence, snap_pos = self.detect_snap_gesture(
                        hand_landmarks.landmark, width, height)
                    
                    if is_snap:
                        self.trigger_snap_effects(snap_pos)
                        
                        if not self.is_invisible:
                            self.is_invisible = True
                            self.invisibility_start_time = time.time()
                        else:
                            self.is_invisible = False
            
            if self.is_invisible:
                elapsed = time.time() - self.invisibility_start_time
                if elapsed >= self.invisibility_duration:
                    self.is_invisible = False
            
            if self.is_invisible and self.background_captured:
                frame = self.apply_invisibility(frame)
            
            if self.snap_effect_active:
                self.update_effects(frame)
            
            if not self.background_captured:
                status = "Press 'C' to capture background"
                color = (0, 0, 255)
            elif self.is_invisible:
                remaining = max(0, self.invisibility_duration - (time.time() - self.invisibility_start_time))
                status = f"INVISIBLE - {remaining:.1f}s"
                color = (0, 255, 255)
            else:
                status = "VISIBLE - Snap to vanish"
                color = (0, 255, 0)
            
            cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, "C - Capture | Q - Quit", (10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Enhanced Thanos Snap', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.capture_background()
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    thanos = EnhancedThanosSnap()
    thanos.run()
