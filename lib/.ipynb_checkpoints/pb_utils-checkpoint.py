import pybullet as p
import numpy as np
import time

pb_joint_indices = np.array([1, 2, 3, 11, 12, 13, 6, 7, 8, 16, 17, 18])

def create_primitives(shapeType=2, rgbaColor=[1, 1, 0, 1], pos = [0, 0, 0], radius = 1, length = 2, halfExtents = [0.5, 0.5, 0.5], baseMass=1, basePosition = [0,0,0]):
    visualShapeId = p.createVisualShape(shapeType=shapeType, rgbaColor=rgbaColor, visualFramePosition=pos, radius=radius, length=length, halfExtents = halfExtents)
    collisionShapeId = p.createCollisionShape(shapeType=shapeType, collisionFramePosition=pos, radius=radius, height=length, halfExtents = halfExtents)
    bodyId = p.createMultiBody(baseMass=baseMass,
                      baseInertialFramePosition=[0, 0, 0],
                      baseVisualShapeIndex=visualShapeId,
                      baseCollisionShapeIndex=collisionShapeId,    
                      basePosition=basePosition,
                      useMaximalCoordinates=True)
    return visualShapeId, collisionShapeId, bodyId

def set_q(q, robot_id, joint_indices,  set_base = False):
    if set_base:
        localInertiaPos = np.array(p.getDynamicsInfo(robot_id,-1)[3])
        q_root = q[0:7]
        ori = q_root[3:]
        Rbase = np.array(p.getMatrixFromQuaternion(ori)).reshape(3,3)
        shift_base = Rbase.dot(localInertiaPos)
        pos = q_root[:3]+shift_base
        p.resetBasePositionAndOrientation(robot_id,pos,ori)
        q_joint = q[7:]
    else:
        q_joint = q
    
    #set joint angles
    for i in range(len(q_joint)):
        p.resetJointState(robot_id, joint_indices[i], q_joint[i])


def vis_traj(qs, vis_func, dt=0.1):
    for q in qs:
        vis_func(q)
        time.sleep(dt)
    
def plot_compare(nrows, ncols, datas, colors, labels, titles=None, filename=None):
    fig,axs = plt.subplots(nrows, ncols)
    fig.set_size_inches(nrows*4, ncols*4)
    D = datas[0].shape[1]
    for i in range(D):
        for j,data in enumerate(datas):
            axs.flatten()[i].plot(data[:,i], colors[j], label=labels[j])
        if titles is not None:
            axs.flatten()[i].set_title(titles[i])
    axs[0,0].legend()
    if filename is not None:
        plt.savefig(filename, dpi=200, facecolor="w")
    plt.show()
    return fig