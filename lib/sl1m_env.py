import numpy as np
import pybullet as p
import pinocchio as pin
import hppfcl
import example_robot_data

def generate_stair_small(stepHeight = 0.1, side_margin = 0.1):
    floor = [[0.16, 1., 0.], [-1.8, 1., 0.], [-1.8, -1., 0.], [0.16, -1., 0.]]

    step1 = [[0.4, 0.6+side_margin, 1*stepHeight], [0.4, -0.16-side_margin, 1*stepHeight], [0.5, -0.16-side_margin, 1*stepHeight], [0.5, 0.6+side_margin, 1*stepHeight]]
    step2 = [[0.75, 0.6+side_margin, 2*stepHeight], [0.75, -0.16-side_margin, 2*stepHeight], [1.7, -0.16-side_margin, 2*stepHeight], [1.7, 0.6+side_margin, 2*stepHeight]]

    afloor = np.array(floor).T
    astep1 = np.array(step1).T
    astep2 = np.array(step2).T

    scene = [[afloor], [astep1], [astep2]]
    return scene

def generate_stair_full(stepHeight=0.1):
    floor = [[0.16, 1., 0.], [-1.8, 1., 0.], [-1.8, -1., 0.], [0.16, -1., 0.]]
    step1 = [[0.35, 0.6, 1*stepHeight], [0.35, -0.16, 1*stepHeight], [0.5, -0.16, 1*stepHeight], [0.5, 0.6, 1*stepHeight]]
    step2 = [[0.65, 0.6, 2*stepHeight], [0.65, -0.16, 2*stepHeight], [0.8, -0.16, 2*stepHeight], [0.8, 0.6, 2*stepHeight]]
    step3 = [[0.95, 0.6, 3*stepHeight], [0.95, -0.16, 3*stepHeight], [1.1, -0.16, 3*stepHeight], [1.1, 0.6, 3*stepHeight]]
    step4 = [[1.25, 0.6, 4*stepHeight], [1.25, -0.16, 4*stepHeight], [2.3, -0.16, 4*stepHeight], [2.3, 0.6, 4*stepHeight]]

    afloor = np.array(floor).T
    astep1 = np.array(step1).T
    astep2 = np.array(step2).T
    astep3 = np.array(step3).T
    astep4 = np.array(step4).T

    scene = [[afloor], [astep1], [astep2], [astep3], [astep4]]
    return scene

def generate_stair_full_wide(stepHeight=0.1, stepWidth=0.3, stepMarginLeft=0.1, stepMarginRight=0.1, finalStepWidth = 1.2):
    floor = [[0.16, 1., 0.], [-1.8, 1., 0.], [-1.8, -1., 0.], [0.16, -1., 0.]]
    step1 = [[0.3 + stepMarginLeft, 0.6, 1*stepHeight], [0.3 + stepMarginLeft, -0.16, 1*stepHeight], [0.3 + 1*stepWidth - stepMarginRight, -0.16, 1*stepHeight], [0.3 + 1*stepWidth - stepMarginRight, 0.6, 1*stepHeight]]
    step2 = [[0.3 + 1*stepWidth + stepMarginLeft, 0.6, 2*stepHeight], [0.3 + 1*stepWidth + stepMarginLeft, -0.16, 2*stepHeight], [0.3 + 2*stepWidth - stepMarginRight, -0.16, 2*stepHeight], [0.3 + 2*stepWidth - stepMarginRight, 0.6, 2*stepHeight]]
    step3 = [[0.3 + 2*stepWidth + stepMarginLeft, 0.6, 3*stepHeight], [0.3 + 2*stepWidth + stepMarginLeft, -0.16, 3*stepHeight], [0.3 + 3*stepWidth - stepMarginRight, -0.16, 3*stepHeight], [0.3 + 3*stepWidth - stepMarginRight, 0.6, 3*stepHeight]]
    step4 = [[0.3 + 3*stepWidth + stepMarginLeft, 0.6, 4*stepHeight], [0.3 + 3*stepWidth + stepMarginLeft, -0.16, 4*stepHeight], [0.3 + 3*stepWidth + finalStepWidth- stepMarginRight, -0.16, 4*stepHeight], [0.3 + 3*stepWidth + finalStepWidth - stepMarginRight, 0.6, 4*stepHeight]]

    afloor = np.array(floor).T
    astep1 = np.array(step1).T
    astep2 = np.array(step2).T
    astep3 = np.array(step3).T
    astep4 = np.array(step4).T

    scene = [[afloor], [astep1], [astep2], [astep3], [astep4]]
    return scene

def setup_pybullet(urdf_name, env_name=None):
    p.resetSimulation()
    p.setTimeStep(0.001)
    p.setGravity(0,0,-9.81)
    #### Plot robot
    robot_id = p.loadURDF(urdf_name)
    plane_id = p.loadURDF('plane.urdf')
    if env_name is not None:
        env_id = p.loadURDF(env_name, useFixedBase=True)
    else:
        env_id = -1
    return robot_id, plane_id, env_id

def load_robot_pinocchio():
    # ANYmal robot model
    robot = example_robot_data.load("anymal")
    rmodel = robot.model
    rmodel.effortLimit *= 0.4
    rmodel.velocityLimit *= 0.5

    # Initial state-
    q0 = rmodel.referenceConfigurations['standing'].copy()
    v0 = pin.utils.zero(rmodel.nv)
    x0 = np.concatenate([q0, v0])

    # Compute initial contact positions
    rdata = rmodel.createData()
    pin.forwardKinematics(rmodel, rdata, q0, v0)
    pin.updateFramePlacements(rmodel, rdata)

    rh_id = rmodel.getFrameId('RH_FOOT')
    rf_id = rmodel.getFrameId('RF_FOOT')
    lh_id = rmodel.getFrameId('LH_FOOT')
    lf_id = rmodel.getFrameId('LF_FOOT')
    
    return rmodel, rdata, q0, v0, rh_id, rf_id, lh_id, lf_id

def plot_obstacles(obstacles):
    for obstacle in obstacles:
        _,_, obs_id = create_primitives(p.GEOM_BOX, halfExtents=obstacle['fullsize']/2, rgbaColor=[0,1,0,1])
        p.resetBasePositionAndOrientation(obs_id, obstacle['pose'].translation, (0,0,0,1))

def create_obstacle_from_scene(scene, stair_height = 0.1, margin_obs = 0.1, actual_radius=0.05, obs_weight =1e2, use_full_size = False, margin_stair = 0.1):
    obstacles = []
    #obstacle_pb_ids = []
    for i in range(1, len(scene)):
        box_center = np.mean(scene[i][0], 1)
        box_center[-1] -= stair_height/2
        box_half_size = (np.max(scene[i][0], 1) - np.min(scene[i][0], 1))/2
        box_half_size[0] += margin_stair
        box_half_size[-1] = stair_height/2

        #create object in pybullet
    #     _,_,box_id = create_primitives(p.GEOM_BOX, halfExtents=box_half_size)
    #     p.resetBasePositionAndOrientation(box_id, box_center, (0,0,0,1))
    #     obstacle_pb_ids.append(box_id)

        #create object in pinocchio
        stair1_pose = pin.SE3.Identity()
        stair1_pose.translation = np.matrix(box_center[:,None])

        stair1 = dict()
        stair1['name'] = 'object'+str(i)
        stair1['fullsize'] = box_half_size*2
        stair1['radius'] = margin_obs #actual radius: 0.05, or half the height
        stair1['actual_radius'] = actual_radius
        if use_full_size:
            stair1['fcl_obj'] = hppfcl.Box(stair1['fullsize'][0], stair1['fullsize'][1], stair1['fullsize'][2]) #fullsize - 2*actual_radius            
        else:
            stair1['fcl_obj'] = hppfcl.Box(stair1['fullsize'][0]-actual_radius*2, stair1['fullsize'][1]-actual_radius*2, stair1['fullsize'][2]-actual_radius*2) #fullsize - 2*actual_radius
        stair1['pose'] = stair1_pose
        stair1['weight'] = obs_weight
        obstacles += [stair1]

    # Define the ground obstacle
    # Ground
    ground_pose = pin.SE3.Identity()
    ground_pose.translation = np.array([0., 0., -1.])  
    ground = dict()
    ground['name'] = 'ground'
    ground['fullsize'] = np.array([10, 10., 2])
    ground['radius'] = 1.03 #actual radius: 1.
    if use_full_size:
        ground['fcl_obj'] = hppfcl.Box(10, 10, 2.) #fullsize - 2*actual_radius        
    else:
        ground['fcl_obj'] = hppfcl.Box(10, 10, 0.) #fullsize - 2*actual_radius
    
    ground['pose'] = ground_pose
    if use_full_size:
        ground['weight'] = 1 * 1e2
    else:
        ground['weight'] = 1 * 1e3
    obstacles += [ground] 
    return obstacles
