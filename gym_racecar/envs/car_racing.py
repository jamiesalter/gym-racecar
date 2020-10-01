"""

Easiest continuous control task to learn from pixels, a top-down racing environment.
Discrete control is reasonable in this environment as well, on/off discretization is
fine.

State consists of STATE_W x STATE_H pixels.

The reward is -0.1 every frame and +1000/N for every track tile visited, where N is
the total number of tiles visited in the track. For example, if you have finished in 732 frames,
your reward is 1000 - 0.1*732 = 926.8 points.

The game is solved when the agent consistently gets 900+ points. The generated track is random every episode.

The episode finishes when all the tiles are visited. The car also can go outside of the PLAYFIELD -  that
is far off the track, then it will get -100 and die.

Some indicators are shown at the bottom of the window along with the state RGB buffer. From
left to right: the true speed, four ABS sensors, the steering wheel position and gyroscope.

To play yourself (it's rather fast for humans), type:

python gym/envs/box2d/car_racing.py

Remember it's a powerful rear-wheel drive car -  don't press the accelerator and turn at the
same time.

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

Modified by Jamie Salter.
"""

import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle

import pyglet
from pyglet import gl

from gym_racecar.envs.car_dynamics import Car

STATE_W = 96   # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 1000

SCALE = 5.0             # Track scale
TRACK_RAD = 600/SCALE   # Track is heavily morphed circle with this radius
PLAYFIELD = 1100/SCALE  # Game over boundary
FPS = 50                # Frames per second
ZOOM = 1                # Camera zoom
ZOOM_FOLLOW = False     # Set to False for fixed view (don't use zoom)

TRACK_DETAIL_STEP = 21/SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 60/SCALE
TRACK_GRASS_WIDTH = TRACK_WIDTH * 2
BORDER = 8/SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]
GRASS_COLOR = [0.4, 0.8, 0.4]

LOOK_AHEAD = 30 # How many tiles the car can 'see' in front of it
VISIBLE_ROAD_COLOR = [0.4, 0.8, 0.4]

class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        obj = {}

        if not u1 or not u2:
            return
        
        # Check for road tiles
        if "road_friction" in u1.__dict__:
            obj['road'] = u1
        elif "road_friction" in u2.__dict__:
            obj['road'] = u2

        # Check for cars
        if "tiles" in u1.__dict__:
            obj['car'] = u1
        elif "tiles" in u2.__dict__:
            obj['car'] = u2

        # Check for grass
        if "grass_idx" in u1.__dict__:
            obj['grass'] = u1
        elif "grass_idx" in u2.__dict__:
            obj['grass'] = u2
        
        if 'road' in obj and 'car' in obj:
            # Change the colour of the road tile to show it's been visited
            obj['road'].color[0] = ROAD_COLOR[0]
            obj['road'].color[1] = ROAD_COLOR[1]
            obj['road'].color[2] = ROAD_COLOR[2]

            if begin:
                # Save the road tile to the `tiles` parameter of the car class
                # The car checks if it's on a tile by checking this set and adjusts it's
                # friction to equal the tiles `road_friction` parameter
                obj['car'].tiles.add(obj['road'])
                if not obj['road'].road_visited:
                    obj['road'].road_visited = True
                    self.env.reward += 1000.0/len(self.env.track)
                    self.env.tile_visited_count += 1
            else:
                obj['car'].tiles.remove(obj['road'])
        
        if 'car' in obj and 'grass' in obj:
            if begin:
                self.env.on_grass_idx.add(obj['grass'].grass_idx)
            else:
                if obj['grass'].grass_idx in self.env.on_grass_idx:
                    self.env.on_grass_idx.remove(obj['grass'].grass_idx)


class CarRacing(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels', 'vector_track'],
        'video.frames_per_second' : FPS
    }

    def __init__(self, verbose=1):
        EzPickle.__init__(self)
        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0,0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.grass = []
        self.on_grass_idx = set()
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.poly = {'grass': [], 'road': [], 'other': []}

        self.action_space = spaces.Box(np.array([-1, 0, 0]),
                                       np.array([+1, +1, +1]),
                                       dtype=np.float32)  # steer, gas, brake

        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

        self.timer = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []

        for t in self.grass:
            self.world.DestroyBody(t)
        self.grass = []

        self.grass_idx = None

        self.car.destroy()

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2*math.pi*c/CHECKPOINTS + self.np_random.uniform(0, 2*math.pi*1/CHECKPOINTS)
            rad = self.np_random.uniform(TRACK_RAD/3, TRACK_RAD)
            if c == 0:
                alpha = 0
                rad = 1.5*TRACK_RAD
            if c == CHECKPOINTS-1:
                alpha = 2*math.pi*c/CHECKPOINTS
                self.start_alpha = 2*math.pi*(-0.5)/CHECKPOINTS
                rad = 1.5*TRACK_RAD
            checkpoints.append((alpha, rad*math.cos(alpha), rad*math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5*TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2*math.pi
            while True: # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2*math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x*dest_dx + r1y*dest_dy  # destination vector projected on rad
            while beta - alpha > 1.5*math.pi:
                 beta -= 2*math.pi
            while beta - alpha < -1.5*math.pi:
                 beta += 2*math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                 beta -= min(TRACK_TURN_RATE, abs(0.001*proj))
            if proj < -0.3:
                 beta += min(TRACK_TURN_RATE, abs(0.001*proj))
            x += p1x*TRACK_DETAIL_STEP
            y += p1y*TRACK_DETAIL_STEP
            track.append((alpha,prev_beta*0.5 + beta*0.5,x,y))
            if laps > 4:
                 break
            no_freeze -= 1
            if no_freeze == 0:
                 break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i-1][0] <= self.start_alpha
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2-i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1:i2-1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x*(track[0][2] - track[-1][2])) +
            np.square(first_perp_y*(track[0][3] - track[-1][3])))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False]*len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i-neg-0][1]
                beta2 = track[i-neg-1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE*0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i-neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i-1]

            # Grass fixtures
            road1_l = (x1 - TRACK_GRASS_WIDTH*math.cos(beta1), y1 - TRACK_GRASS_WIDTH*math.sin(beta1))
            road1_r = (x1 + TRACK_GRASS_WIDTH*math.cos(beta1), y1 + TRACK_GRASS_WIDTH*math.sin(beta1))
            road2_l = (x2 - TRACK_GRASS_WIDTH*math.cos(beta2), y2 - TRACK_GRASS_WIDTH*math.sin(beta2))
            road2_r = (x2 + TRACK_GRASS_WIDTH*math.cos(beta2), y2 + TRACK_GRASS_WIDTH*math.sin(beta2))
            self.poly['grass'].append(( [road1_l, road1_r, road2_r, road2_l], GRASS_COLOR ))
            t = self.world.CreateStaticBody(fixtures=fixtureDef(
                shape=polygonShape(vertices=[road1_l, road1_r, road2_r, road2_l]), isSensor = True))
            t.userData = t
            t.userData.grass_idx = len(self.grass)
            self.grass.append(t)

            # Road fixtures
            road1_l = (x1 - TRACK_WIDTH*math.cos(beta1), y1 - TRACK_WIDTH*math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH*math.cos(beta1), y1 + TRACK_WIDTH*math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH*math.cos(beta2), y2 - TRACK_WIDTH*math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH*math.cos(beta2), y2 + TRACK_WIDTH*math.sin(beta2))
            t = self.world.CreateStaticBody(fixtures=fixtureDef(
                shape=polygonShape(vertices=[road1_l, road1_r, road2_r, road2_l]), isSensor = True))
            t.userData = t
            t.road_visited = False
            t.road_friction = 1.0
            # Vary the colour of the road
            c = 0.01*(i%3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            self.poly['road'].append(( [road1_l, road1_r, road2_r, road2_l], t.color ))
            self.road.append(t)

            # Red/white borders (only occur in the render)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
                b1_r = (x1 + side * (TRACK_WIDTH+BORDER) * math.cos(beta1),
                        y1 + side * (TRACK_WIDTH+BORDER)*math.sin(beta1))
                b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
                b2_r = (x2 + side * (TRACK_WIDTH+BORDER) * math.cos(beta2),
                        y2 + side * (TRACK_WIDTH+BORDER) * math.sin(beta2))
                self.poly['other'].append(([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0)))
        
        self.track = track
        return True

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.poly = {'grass': [], 'road': [], 'other': []}
        self.on_grass_idx.clear()

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print("retry to generate track (normal if there are not many instances of this message)")
        self.car = Car(self.world, *self.track[0][1:4])

        return self.step(None)[0]

    def step(self, action):
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        self.state = self.render("state_pixels")

        step_reward = 0
        done = False
        if action is not None: # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track):
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD or len(self.on_grass_idx) == 0:
                done = True
                step_reward = -100

        return self.state, step_reward, done, {}

    def render(self, mode='human'):
        assert mode in ['human', 'state_pixels', 'rgb_array', 'vector_track']

        # Note that the verticies are arranged in order of how close each tile is.
        # But there is no guarantee in the order of which side (left or right) is returned.
        # And the first tile returned may include the both right and then both left vertices of the quad
        visible_road_vertices = self.getRoadVertices()

        #if self.car is not None and self.car.hull is not None:
            #local_v = np.array([self.car.hull.GetLocalPoint(v) for v in visible_road_vertices])
            #local_v = np.c_[local_v, np.linalg.norm(local_v, axis=1)]
            #if self.timer % 20 == 0:
            #    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            #    print(local_v)
            #self.timer += 1

        if mode == "vector_track":

            if self.car is None or self.car.hull is None:
                return None
            
            local_v = np.array([self.car.hull.GetLocalPoint(v) for v in visible_road_vertices])
        
            # Return two vectors of each side of the track relative to the frame of the car
            # Also return the following wheel parameters: steer, gas, brake, speed and vehicle speed (forward and sideways)
            arr = np.array([self.car.wheels[0].steer, 
                            self.car.wheels[2].gas,
                            self.car.wheels[0].brake,
                            self.car.wheels[0].vr, self.car.wheels[1].vr,
                            self.car.wheels[2].vr, self.car.wheels[3].vr,
                            self.car.wheels[2].vf, self.car.wheels[2].vs])
            arr = np.r_[arr, local_v[:,0], local_v[:,1]]
            return arr

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                x=20, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
            self.transform = rendering.Transform()
            if not ZOOM_FOLLOW:
                zoom = WINDOW_H / (2*PLAYFIELD)
                self.transform.set_scale(zoom, zoom)
                self.transform.set_translation(WINDOW_W/2,WINDOW_H/2)
                self.transform.set_rotation(0)

        if "t" not in self.__dict__: return  # reset() not called yet

        if ZOOM_FOLLOW:
            # Zoom starts at 0.1*SCALE and ends at ZOOM*SCALE
            zoom = 0.1*SCALE*max(1-self.t, 0) + ZOOM*SCALE*min(self.t, 1)   # Animate zoom first second
            scroll_x = self.car.hull.position[0]
            scroll_y = self.car.hull.position[1]
            angle = -self.car.hull.angle
            vel = self.car.hull.linearVelocity
            if np.linalg.norm(vel) > 0.5:
                angle = math.atan2(vel[0], vel[1])
            self.transform.set_scale(zoom, zoom)
            self.transform.set_translation(
                WINDOW_W/2 - (scroll_x*zoom*math.cos(angle) - scroll_y*zoom*math.sin(angle)),
                WINDOW_H/4 - (scroll_x*zoom*math.sin(angle) + scroll_y*zoom*math.cos(angle)))
            self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        # scale the viewport to output at different sizes for ML
        win.clear()
        t = self.transform
        if mode == 'rgb_array':
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == 'state_pixels':
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            # an NSOpenGLContext seems to be something in Mac
            # The following line is trying to scale based on Mac's scaling settings
            if hasattr(win.context, '_nscontext'):
                pixel_scale = win.context._nscontext.view().backingScaleFactor()  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road(visible_road_vertices)
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == 'human':
            win.flip()
            return self.viewer.isopen

        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def getRoadVertices(self):
        vertices = []
        vertices2 = []

        if self.road is not None and len(self.on_grass_idx) > 0:
            tile_idx = max(self.on_grass_idx)
            for i, t in enumerate(self.road):
                if tile_idx is None:
                    break
                if i >= tile_idx and i < tile_idx + LOOK_AHEAD:
                    vertices.extend(t.fixtures[0].shape.vertices)
                if i < (tile_idx + LOOK_AHEAD) - len(self.road):
                    # We're adding this separately as we need to ensure these get added to the end of the list
                    vertices2.extend(t.fixtures[0].shape.vertices)

        vertices.extend(vertices2)

        # Remove duplicates in the road vertices
        # This works because dictionaries can't have duplicate keys
        vertices = list(dict.fromkeys(vertices))

        return vertices

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self, visible_vertices):
        gl.glBegin(gl.GL_QUADS)

        # Draw background
        gl.glColor4f(0.1, 0.1, 0.1, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)

        # Draw grass
        if len(self.on_grass_idx) == 0:
           grass_idx = None # this only occurs just before a reset when you drive off the track
        else: 
            grass_idx = max(self.on_grass_idx)
        for i, (poly, color) in enumerate(self.poly['grass']):
            if grass_idx is not None and ((i >= grass_idx and i < grass_idx + LOOK_AHEAD) or i < (grass_idx + LOOK_AHEAD)-len(self.poly['grass'])):
                color = VISIBLE_ROAD_COLOR
            gl.glColor4f(*color, 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)

        # Draw road and other items (like the red/white borders)
        for key in ['road', 'other']:
            for poly, color in self.poly[key]:
                gl.glColor4f(*color, 1)
                for p in poly:
                    gl.glVertex3f(p[0], p[1], 0)

        gl.glEnd()

        # Draw the visible vertices
        gl.glPointSize(5)
        gl.glBegin(gl.GL_POINTS)
        for i, v in enumerate(visible_vertices):
            gl.glColor4f(1-i/len(visible_vertices), 0, 0, 1.0)
            gl.glVertex3f(v[0], v[1], 0)
        gl.glEnd()

    def render_indicators(self, W, H):
        gl.glBegin(gl.GL_QUADS)
        s = W/40.0
        h = H/40.0
        gl.glColor4f(0, 0, 0, 1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5*h, 0)
        gl.glVertex3f(0, 5*h, 0)
        gl.glVertex3f(0, 0, 0)

        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h, 0)
            gl.glVertex3f((place+0)*s, h, 0)

        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, 4*h , 0)
            gl.glVertex3f((place+val)*s, 4*h, 0)
            gl.glVertex3f((place+val)*s, 2*h, 0)
            gl.glVertex3f((place+0)*s, 2*h, 0)

        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))
        vertical_ind(5, 0.02*true_speed, (1, 1, 1))
        vertical_ind(7, 0.01*self.car.wheels[0].omega, (0.0, 0, 1)) # ABS sensors
        vertical_ind(8, 0.01*self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01*self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10,0.01*self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0*self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8*self.car.hull.angularVelocity, (1, 0, 0))
        gl.glEnd()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()


if __name__=="__main__":
    from pyglet.window import key
    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xff0d: restart = True
        if k == key.LEFT:  a[0] = -1.0
        if k == key.RIGHT: a[0] = +1.0
        if k == key.UP:    a[1] = +1.0
        if k == key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT  and a[0] == -1.0: a[0] = 0
        if k == key.RIGHT and a[0] == +1.0: a[0] = 0
        if k == key.UP:    a[1] = 0
        if k == key.DOWN:  a[2] = 0
    env = CarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor
        env = Monitor(env, '/tmp/video-test', force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()
