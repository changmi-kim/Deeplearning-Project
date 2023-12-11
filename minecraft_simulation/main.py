from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController

app = Ursina()

jump_height = 4 # Default: 2
jump_duration = 0.5 # Default: 0.5
jump_fall_after = 0.35 # Default: 0.35
gravity_scale = 1 # Default: 1
mouse_sensitivity = Vec2(40,40) # Default: (40,40)
run_speed = 5 # Default: 5

window.fps_counter.enabled = False
window.exit_button.visible = False

punch = Audio('assets/punch', autoplay=False)

blocks = [
    load_texture('assets/grass.png'), # 0
    load_texture('assets/grass.png'), # 1
    load_texture('assets/stone.png'), # 2
    load_texture('assets/gold.png'),  # 3
    load_texture('assets/lava.png'),  # 4
    load_texture('assets/pinklab.png'),  # 5
]

block_id = 1

def input(key):
    global block_id, hand
    if key.isdigit():
        block_id = int(key)
        if block_id >= len(blocks):
            block_id = len(blocks) - 1
        hand.texture = blocks[block_id]

sky = Entity(
    parent=scene,
    model='sphere',
    texture=load_texture('assets/sky.jpg'),
    scale=500,
    double_sided=True
)

hand = Entity(
    parent=camera.ui,
    model='assets/block',
    texture=blocks[block_id],
    scale=0.2,
    rotation=Vec3(-10, -10, 10),
    position=Vec2(0.6, -0.6)
)

def update():
    if held_keys['left mouse'] or held_keys['right mouse']:
        punch.play()
        hand.position = Vec2(0.4, -0.5)
    else:
        hand.position = Vec2(0.6, -0.6)

class Voxel(Button):
    def __init__(self, position=(0, 0, 0), texture='assets/grass.png'):
        super().__init__(
            parent=scene,
            position=position,
            model='assets/block',
            origin_y=0.5,
            texture=texture,
            color=color.color(0, 0, random.uniform(0.9, 1.0)),
            scale=0.5
        )

    def input(self, key):
        if self.hovered:
            if key == 'left mouse down':
                Voxel(position=self.position + mouse.normal, texture=blocks[block_id])
            elif key == 'right mouse down':
                destroy(self)

for z in range(16):
    for x in range(16):
        voxel = Voxel(position=(x+1, 1, z-8))

new_voxel = Voxel(position=(20, 10, -4), texture='assets/pinklab.png')
new_voxel = Voxel(position=(0, 0, 0), texture='assets/stone.png')

# 시간을 추적할 변수
elapsed_time = 0
timer_text = Text(text='Time: 0', origin=(0, -6), color=color.white, scale=3)

def update():
    global elapsed_time
    # 프레임마다 시간 업데이트
    elapsed_time += time.dt
    timer_text.text = f'Time: {int(elapsed_time)}'

#################
player = FirstPersonController()

player.jump_height = jump_height
player.jump_up_duration = jump_duration
player.mouse_sensitivity = mouse_sensitivity
player.speed = run_speed
player.gravity = gravity_scale

app.run()
