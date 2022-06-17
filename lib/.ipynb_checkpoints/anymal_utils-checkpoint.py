from caracal.trajectory import SwingFootTrajectoryGenerator
from caracal.contact import ContactSchedule, ContactPhase
from caracal.gaits import QuadrupedalGaitGenerator
import pinocchio
import numpy as np
import copy
from ocp import *
import pybullet as p

contact_dict = {'LF': 'LF_FOOT', 'RF':'RF_FOOT', 'LH':'LH_FOOT', 'RH':'RH_FOOT'}
contact_names = ['LF_FOOT', 'LH_FOOT', 'RF_FOOT', 'RH_FOOT']
contact_indices = {'DS': 0, 'LF': 1, 'LH':2, 'RF':3,  'RH':4}

class Simulator():
    def __init__(self, mpc, plane_id, robot_id, rmodel, timestep = 0.001):
        self._mpc = mpc
        self._q = np.zeros(19)
        self._v = np.zeros(18)
        self._plane_id = plane_id
        self._robot_id = robot_id
        self._forceSensors = ["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"]
        self.i = 0
        self._xref = self._mpc.get_xopt()
        self._uff = self._mpc.get_uopt()
        self._Kref = self._mpc.get_Kopt()
        self._fref = self._mpc.get_fopt()
        self._sref = self._mpc.get_sopt()
        self._timestep = timestep
        self._model = rmodel
        self._data = rmodel.createData()
        self._parseRobot()
        
        self._dt = timestep  # IMPORTANT: use the period of the controller loop
        self._state = crocoddyl.StateMultibody(self._model)
        self._actuation = crocoddyl.ActuationModelFloatingBase(self._state)
        costs = crocoddyl.CostModelSum(self._state, self._actuation.nu)
        contacts = crocoddyl.ContactModelMultiple(self._state, self._actuation.nu)
        self._baumgarteGains = np.array([0., 50.])

        contact3DNames = ["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"]
        for name in contact3DNames:
            frameId = self._state.pinocchio.getFrameId(name)
            contact3D = crocoddyl.ContactModel3D(self._state, frameId, np.zeros(3), self._actuation.nu, self._baumgarteGains)
            contacts.addContact(name, contact3D)
#         contact6DNames = []
#         for name in contact6DNames:
#             frameId = state.pinocchio.getFrameId(name)
#             contact6D = crocoddyl.ContactModel6D(state, frameId, pinocchio.SE3.Identity(), actuation.nu,
#                                                  baumgarteGains)
#             contacts.addContact(name, contact6D)

        diffModel = crocoddyl.DifferentialActionModelContactFwdDynamics(self._state, self._actuation, contacts, costs, 0., False)
        self._amodel = crocoddyl.IntegratedActionModelEuler(diffModel, self._dt)
        self._adata = self._amodel.createData()
        
        
        
    def resetRobot(self, q0):
        set_q_std(q0)
        self.i = 0
        
    def _parseRobot(self):
        # Getting the dictionary that maps the bullet non-fixed joint names to indexes
        self._jointsBullet = dict()
        for j in range(p.getNumJoints(self._robot_id)):
            joint_info = p.getJointInfo(self._robot_id, j)
            if joint_info[2] != p.JOINT_FIXED:
                self._jointsBullet[joint_info[1].decode("utf-8")] = j
        
    def _updateActualState(self):
        # Update the base state
        self._q[:3], self._q[3:7] = p.getBasePositionAndOrientation(self._robot_id)
        R = pinocchio.Quaternion(self._q[3:7]).toRotationMatrix()
        self._v[:3], self._v[3:6] = p.getBaseVelocity(self._robot_id)
        self._v[:3] = np.dot(R.T, self._v[:3])
        self._v[3:6] = np.dot(R.T, self._v[3:6])

        self._f = {name: [0, pinocchio.Force.Zero(), 1] for name in self._forceSensors}
        self._s = {name: [np.zeros(3), 0.] for name in self._forceSensors}
        contact_list = p.getContactPoints()
        for contact in contact_list:
            force_n = contact[9]
            if force_n > 0.:
                force_1 = contact[10]
                force_2 = contact[12]
                surface_n = -1 * np.array(contact[7])
                surface_1 = -1 * np.array(contact[11])
                surface_2 = -1 * np.array(contact[13])
                force = force_n * surface_n + force_1 * surface_1 + force_2 * surface_2
                friction_mu = p.getDynamicsInfo(self._plane_id, -1)[1]
                if contact[4] == -1:
                    name = p.getJointInfo(self._robot_id, contact[3])[12].decode("utf-8")
                    if name in self._forceSensors:
                        self._f[name] = [
                            0, pinocchio.Force(-force, np.zeros(3)), 2 if np.linalg.norm(force) > 0. else 1
                        ]
                        self._s[name] = [-surface_n, friction_mu]
                else:
                    name = p.getJointInfo(self._robot_id, contact[4])[12].decode("utf-8")
                    if name in self._forceSensors:
                        self._f[name] = [
                            0, pinocchio.Force(force, np.zeros(3)), 2 if np.linalg.norm(force) > 0. else 1
                        ]
                        self._s[name] = [surface_n, friction_mu]
                        
        # Update the joint state
        for j in range(2, self._model.njoints):
            name = self._model.names[j]
            idx = self._jointsBullet[name]
            state = p.getJointState(self._robot_id, idx)
            self._q[j + 5] = state[0]
            self._v[j + 4] = state[1]

        # Update the contact position
        self._p = dict()
        self._pd = dict()
        for name in self._f.keys():
            pinocchio.forwardKinematics(self._model, self._data, self._q, self._v)
            frame_id = self._model.getFrameId(name)
            oMf = pinocchio.updateFramePlacement(self._model, self._data, frame_id)
            ovf = pinocchio.getFrameVelocity(self._model, self._data, frame_id,
                                             pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            self._p[name] = oMf
            self._pd[name] = ovf
            
    def _setJointCommand(self,u):
        # This resets the velocity controller for joints that are not actuated
        for name in self._jointsBullet:
            idx = self._jointsBullet[name]
            p.setJointMotorControl2(self._robot_id, idx, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        # Sending the joint torque commands to bullet engine
        for j in range(2, self._model.njoints):
            name = self._model.names[j]
            idx = self._jointsBullet[name]

            # Sending the torque commands to the bullet engine
            p.setJointMotorControl2(self._robot_id, idx, p.TORQUE_CONTROL, force=u[j - 2])
            
    def _computeContactDynamics(self, x, u, f):
        # Enable active contacts based on a force threshold
        for name in self._amodel.differential.contacts.contacts.todict().keys():
            self._amodel.differential.contacts.changeContactStatus(name, False)
        for name, force in f.items():
            if np.linalg.norm(force[1].vector) > 0.:
                self._amodel.differential.contacts.changeContactStatus(name, True)
        self._amodel.calc(self._adata, x, u)
        return self._adata.xnext
    

    def update(self, q, v, f, xs_0=None, us_0=None):
        if self.i == 10:
            self._mpc.step(q, v, f, xs_0=None, us_0=None)
            self._xref = self._mpc.get_xopt()
            self._uff = self._mpc.get_uopt()
            self._Kref = self._mpc.get_Kopt()
            self._fref = self._mpc.get_fopt()
            for name, force in self._fref.items():
                force.append(2 if np.linalg.norm(force[1]) > 0. else 1)
            self._sref = self._mpc.get_sopt()
            self.i = 1
        else:
            self.i += 1

        self._xref = self._computeContactDynamics(self._xref, self._uff, f)
        u_fb = np.dot(self._Kref, self._mpc._state.diff(np.hstack([q, v]), self._xref))
        u = self._uff + u_fb
        return u, self._xref[:self._model.nq], self._xref[self._model.nq:], self._fref, self._sref
    
    def step(self, xs_0=None, us_0=None):
        self._updateActualState()
        tau, q_des, v_des, f_des, s_des = self.update(
        copy.deepcopy(self._q), copy.deepcopy(self._v), copy.deepcopy(self._f), xs_0, us_0)
        self._mpc._solver.computeDynamicFeasibility()
        feas = self._mpc._solver.ffeas

        self._setJointCommand(tau)
        p.stepSimulation()
        time.sleep(self._timestep)
        return copy.deepcopy(np.concatenate([self._q, self._v])), copy.deepcopy(tau), feas, copy.deepcopy(self._mpc._solver.xs), copy.deepcopy(self._mpc._solver.us)

def create_lqt_init(lin_sys, q_traj, x0, x_target, T, Q_w = 100):
    '''
    compute xs and us from the predicted qs (position traj) via LQT
    '''
    if len(x0) == lin_sys.Du:
        x0 = np.concatenate([x0, np.zeros(lin_sys.Du)])
    if len(x_target) == lin_sys.Du:
        x_target = np.concatenate([x_target, np.zeros(lin_sys.Du)])
    
    lin_sys.set_init_state(x0)
    lqt = finiteLQT(lin_sys)
    
    #set lqt parameters
    Q = np.identity(lin_sys.Dx)*Q_w
    Q[lin_sys.Du:,lin_sys.Du:] *= 0

    Qf = np.identity(lin_sys.Dx)*Q_w
    Qf[lin_sys.Du:,lin_sys.Du:] *= 0
#     Qf[3:,3:] *= 0

    R = np.identity(lin_sys.Du)*0.001

    x_ref = np.concatenate([q_traj, np.zeros((q_traj.shape))], axis=1)
    
    #set and solve lqt
    lqt.set_ref(x_ref)
    lqt.set_timestep(T)
    lqt.set_cost(Q, R, Qf)
    xs_init, us_init = lqt.solve()
    return xs_init, us_init

def pad_vector(vec, length):
    '''
    Pad a vector to the desired length by adding zero(s)
    '''
    if len(vec) == length:
        return vec
    else:
        add_vec = np.zeros(length-len(vec))
        return np.concatenate([vec, add_vec])


def extract_mpc_contact_sequence(t0, x0, contact_phases, target_contact_poses, num_of_contact_pose, MPC_HORIZON, rmodel, rdata, total_phase=16, total_pose=36):
    '''
    Based on the current time, extract the next N contact phases & current interval & contact poses
    '''
    init_pose = compute_foot_pose(rmodel, rdata, x0[:19])
    contact_pose = np.array(init_pose)
    mpc_phases = []
    prev_phase = np.ones(4)*(-1)
    first_interval = 0
    
    for i in range(MPC_HORIZON):    
        t = t0 + i
        cur_phase = contact_phases[:,t]
        if not np.allclose(cur_phase, prev_phase):
            mpc_phases.append(cur_phase)
            prev_phase = cur_phase
            if len(mpc_phases) == 2:
                #we reach the end of the first phase, so the time interval is i
                first_interval = i
            #get the target contact pose
            for j in range(4):
                if cur_phase[j] == 0:
                    contact_pose = np.concatenate([contact_pose, target_contact_poses[j*3:(j+1)*3, t]])
            if np.sum(cur_phase) == 4:
                #double support, there is no moving foot
                #contact pose: set as zero
                contact_pose = np.concatenate([contact_pose, np.zeros(num_of_contact_pose)])

    #mpc_phase, first_interval, contact_pose
#     print(len(mpc_phases), len(contact_pose))
    mpc_phase = pad_vector(np.concatenate(mpc_phases), total_phase)
    contact_pose = pad_vector(contact_pose, total_pose)
    return mpc_phase, first_interval, contact_pose


# def extract_mpc_contact_sequence(t0, x0, contact_phases, target_contact_poses, num_of_contact_pose, MPC_HORIZON, rmodel,
#                                  rdata, total_phase=16, total_pose=36):
#     '''
#     Based on the current time, extract the next N contact phases & current interval & contact poses
#     '''
#     init_pose = compute_foot_pose(rmodel, rdata, x0[:19])
#     contact_pose = np.array(init_pose)
#     mpc_phases = []
#     prev_phase = np.ones(4) * (-1)
#     first_interval = 0
#
#     for i in range(MPC_HORIZON):
#         t = t0 + i
#         cur_phase = contact_phases[:, t]
#         if not np.allclose(cur_phase, prev_phase):
#             mpc_phases.append(cur_phase)
#             prev_phase = cur_phase
#             if len(mpc_phases) == 2:
#                 first_interval = i
#             # get the target contact pose
#             for j in range(4):
#                 if cur_phase[j] == 0:
#                     contact_pose = np.concatenate([contact_pose, target_contact_poses[j * 3:(j + 1) * 3, t]])
#             if np.sum(cur_phase) == 4:
#                 # double support, there is no moving foot
#                 # contact pose: set as zero
#                 contact_pose = np.concatenate([contact_pose, np.zeros(num_of_contact_pose)])
#
#     # mpc_phase, first_interval, contact_pose
#     mpc_phase = pad_vector(np.concatenate(mpc_phases), total_phase)
#     contact_pose = pad_vector(contact_pose, total_pose)
#     return mpc_phase, first_interval, contact_pose


def compute_foot_pose(rmodel, rdata, q):
    '''
    Compute foot pose, given q
    '''
    pinocchio.forwardKinematics(rmodel, rdata, q)
    pinocchio.updateFramePlacements(rmodel, rdata)
    
    lf_pin_id = rmodel.getFrameId(contact_names[0])
    lh_pin_id = rmodel.getFrameId(contact_names[1])
    rf_pin_id = rmodel.getFrameId(contact_names[2])
    rh_pin_id = rmodel.getFrameId(contact_names[3])
    lf_pos = rdata.oMf[lf_pin_id].translation
    lh_pos = rdata.oMf[lh_pin_id].translation
    rf_pos = rdata.oMf[rf_pin_id].translation
    rh_pos = rdata.oMf[rh_pin_id].translation
    pose = np.concatenate([lf_pos, lh_pos, rf_pos, rh_pos])
    return pose

def extract_contact_poses(gait, c0, MPC_HORIZON=85):
    '''
    Given the gait, extract the contact poses at each time step from the gait
    '''

    contact_poses = np.zeros((12, gait.T + MPC_HORIZON))
    target_contact_poses = np.zeros((12, gait.T + MPC_HORIZON))
    
    for i, leg_phases in enumerate(gait.phases):
        t = 0
        prev_pose = np.copy(c0[gait.contactNames[i]].translation)
        for phase in leg_phases:
            T = phase.T
            try:
                if phase.trajectory is not None:
                    #swing phase
                    for j in range(phase.T):
                        contact_poses[i*3:(i+1)*3, t + j] = phase.trajectory.position(j).translation
                    prev_pose = phase.trajectory.endPosition().translation

            except:
                #stance phase
                contact_poses[i*3:(i+1)*3, t:t+T] = np.array([prev_pose]*T).T
            target_contact_poses[i*3:(i+1)*3,t:t+T] = np.array([prev_pose]*T).T
            t += T
    prev_pose = contact_poses[:,gait.T-1]
    contact_poses[:, gait.T : gait.T+MPC_HORIZON] = np.array([prev_pose]*MPC_HORIZON).T
    target_contact_poses[:, gait.T : gait.T+MPC_HORIZON] = np.array([prev_pose]*MPC_HORIZON).T
    
    return contact_poses, target_contact_poses

def extract_contact_poses_from_poses(gait, foot_poses, MPC_HORIZON=85):
    '''
    Given the gait, extract the contact poses at each time step from the given sequence of contact poses
    '''

    contact_poses = np.zeros((12, gait.T + MPC_HORIZON))
    target_contact_poses = np.zeros((12, gait.T + MPC_HORIZON))
    
    for i, leg_phases in enumerate(gait.phases):
        t = 0
        for phase in leg_phases:
            T = phase.T
            for j in range(phase.T):
                contact_poses[i*3:(i+1)*3, t + j] = foot_poses[i*3:(i+1)*3, t + j]
            target_pose = foot_poses[i*3:(i+1)*3, t + phase.T-1]
            target_contact_poses[i*3:(i+1)*3,t:t+T] = np.array([target_pose]*T).T
            t += T
    prev_pose = contact_poses[:,gait.T-1]
    contact_poses[:, gait.T : gait.T+MPC_HORIZON] = np.array([prev_pose]*MPC_HORIZON).T
    target_contact_poses[:, gait.T : gait.T+MPC_HORIZON] = np.array([prev_pose]*MPC_HORIZON).T
    
    return contact_poses, target_contact_poses



def extract_contact_phase(gait, MPC_HORIZON = 85):
    '''
    Given the gait, extract the contact phases at each time step
    '''

    contact_phases = np.ones((len(gait.contactNames), gait.T + MPC_HORIZON)).astype(np.int32)

    for i in range(len(gait.phases)):
        leg_phases= gait.phases[i]
        t = 0
        for phase in leg_phases:
            T = phase.T
            try:
                if phase.trajectory is not None:
                    contact_phases[i, t:t+T] *= 0
            except Exception:
                contact_phases[i, :t+T] *= 1
            t += T
    return contact_phases

def copy_from_mpc(mpc):
    '''
    Copy important data from MPC, so we can repeat the step function reliably
    '''

    runningModels = []
    runningDatas = []
    terminalModel = []
    terminalData = []

    for i in range(len(mpc._problem.runningModels)):
        runningDatas.append(mpc._problem.runningDatas[i])
        runningModels.append(mpc._problem.runningModels[i])

    terminalModel.append(mpc._problem.terminalModel)
    terminalData.append(mpc._problem.terminalData)

    nonRunningModels = []
    nonRunningDatas = []

    for i in range(len(mpc._nonrunningModels)):
        nonRunningModels.append(mpc._nonrunningModels[i])
        nonRunningDatas.append(mpc._nonrunningDatas[i])

    switchModels = []
    switchDatas = []
    for i in range(len(mpc._switchModels)):
        switchModels.append(mpc._switchModels[i])
        switchDatas.append(mpc._switchDatas[i])
    
    queue_cs = copy.deepcopy(mpc._queue_cs)
    cur_timeline = mpc._timeline 
    reg = mpc._reg
    return runningModels, runningDatas, terminalModel, terminalData, nonRunningModels, nonRunningDatas, switchModels, switchDatas, queue_cs, cur_timeline, reg

def copy_to_mpc(mpc, runningModels, runningDatas, terminalModel, terminalData, nonRunningModels, nonRunningDatas, switchModels, switchDatas, queue_cs, cur_timeline, reg):
    '''
    Copy important data to MPC, so we can repeat the step function reliably
    '''

    for i in range(len(mpc._problem.runningModels)):
        mpc._problem.updateNode(i, runningModels[i], runningDatas[i] )

    mpc._problem.updateNode(mpc._problem.T, terminalModel[0], terminalData[0])

    mpc._nonrunningModels.clear()
    mpc._nonrunningDatas.clear()
    for i in range(len(nonRunningModels)):
        mpc._nonrunningModels.append(nonRunningModels[i])
        mpc._nonrunningDatas.append(nonRunningDatas[i])

    mpc._switchModels.clear()
    mpc._switchDatas.clear()
    
    for i in range(len(switchModels)):
        mpc._switchModels.append(switchModels[i])
        mpc._switchDatas.append(switchDatas[i])
        
    mpc._queue_cs = copy.deepcopy(queue_cs)
    mpc._timeline = cur_timeline
    mpc._reg = reg


def generate_contact_phase(name,contact_init, contact_target, dt=0.01, N_ds=5, N_ss=20, N_start_ds = 20, N_end_ds = 20, stepHeight=0.15, startPhase=False, endPhase=False, S = 4, ws = None):
    if startPhase:
        N_0 = N_start_ds
        N = N_ss + N_ds + N_start_ds
        N_T = N_ds
    elif endPhase:
        N_0 = 0
        N = N_ss +  N_end_ds
        N_T = N_end_ds
    else:
        N = N_ss + N_ds
        N_0 = 0
        N_T = N_ds
        
    gait = ContactSchedule(dt, N, S, contact_names)
    
    #If there is a footswing given
    contact_names_cur = list(contact_names)
    if name != '':
        swingTraj = SwingFootTrajectoryGenerator(dt, N_ss, stepHeight, contact_init , contact_target , ws)
        contact_names_cur.remove(name)
        gait.addSchedule(
        name, [
            ContactPhase(N_0),
            ContactPhase(N_ss, trajectory=swingTraj),
            ContactPhase(N_T),
        ]
        )
    
    for contact_name in contact_names_cur:
        gait.addSchedule(
        contact_name, [
            ContactPhase(N)
        ]
        )
        
    return gait

# def generate_gait(gait_sequence, contact_sequence, N_ds = 10, N_ss = 30):
#     gait_sequence = list(gait_sequence)
#     for i in range(gait_sequence.count('DS')): gait_sequence.remove('DS')

#     generated_gait = []
#     for i,gait_name in enumerate(gait_sequence):
#         if i == 0: 
#             start_phase = True
#             end_phase = False
#         elif i == len(gait_sequence)-1:
#             start_phase = False
#             end_phase = True
#         else:
#             start_phase = False
#             end_phase = False
            
            
#         c0 = pinocchio.SE3(np.eye(3), contact_sequence[i][contact_dict[gait_name]])
#         c1 = pinocchio.SE3(np.eye(3),  contact_sequence[i+1][contact_dict[gait_name]])
#         gait = generate_contact_phase(contact_dict[gait_name], c0, c1, N_ds = N_ds, N_ss = N_ss, startPhase=start_phase, endPhase=end_phase)
#         generated_gait += [gait]

#     gait = generated_gait[0]
#     for gait_i in generated_gait[1:]:
#         gait += gait_i
        
#     return gait
       
def create_contact_sequence_slim_to_caracal(rh_poses, rf_poses, lh_poses, lf_poses):
    '''
    Given sl1m sequence of contact poses, generate the contact sequence in caracal format (cs)
    '''
    N = len(rh_poses)-1

    cs = []
    for i in range(N):
        cs0 = dict()
        if rh_poses[i] is not None:
            cs0["RH_FOOT"] = pinocchio.SE3(np.eye(3), rh_poses[i])
        else:
            cs0["RH_FOOT"] = pinocchio.SE3(np.eye(3), rh_poses[i+1])

        if rf_poses[i] is not None:
            cs0["RF_FOOT"] = pinocchio.SE3(np.eye(3), rf_poses[i])
        else:
            cs0["RF_FOOT"] = pinocchio.SE3(np.eye(3), rf_poses[i+1])

        if lh_poses[i] is not None:
            cs0["LH_FOOT"] = pinocchio.SE3(np.eye(3), lh_poses[i])
        else:
            cs0["LH_FOOT"] = pinocchio.SE3(np.eye(3), lh_poses[i+1])

        if lf_poses[i] is not None:
            cs0["LF_FOOT"] = pinocchio.SE3(np.eye(3), lf_poses[i])
        else:
            cs0["LF_FOOT"] = pinocchio.SE3(np.eye(3), lf_poses[i+1])
        cs.append(cs0)
    return cs    

def create_gait_trot(cs, N_ds = 45, N_ss = 30, stepHeight = 0.1, ws = None):
    gait_generator = QuadrupedalGaitGenerator()
    N = len(cs)
    for i in range(N-1):
        if i == 0:
            gait = gait_generator.trot([cs[i], cs[i+1]], N_ds, N_ss, 0, 0, stepHeight, True, False, ws)
        else:
            gait += gait_generator.trot([cs[i], cs[i+1]], N_ds, N_ss, 0, 0, stepHeight, False, False, ws)
    return gait

    
def get_contact_status(gait):
    contact_status = []
    for c in range(gait.C):
        contact_status_i = []
        for i,phase in enumerate(gait.phases[c]):
            if i%2 == 0:
                contact_status_i += list(np.ones(phase.T))
            else:
                contact_status_i += list(np.zeros(phase.T))
        contact_status += [contact_status_i]

    contact_status = np.array(contact_status).T
    return contact_status

def get_com_contactposes_traj(rmodel, rdata, ddp):
    lf_pin_id = rmodel.getFrameId(contact_names[0])
    lh_pin_id = rmodel.getFrameId(contact_names[1])
    rf_pin_id = rmodel.getFrameId(contact_names[2])
    rh_pin_id = rmodel.getFrameId(contact_names[3])
    dof = int(ddp.problem.nx/2)+1
    v0 = np.zeros(dof-1)
    contact_poses = []
    cs = []
    for x in ddp.xs:
        pinocchio.forwardKinematics(rmodel, rdata, x[:dof], v0)
        pinocchio.updateFramePlacements(rmodel, rdata)
        lf_pos = rdata.oMf[lf_pin_id].translation
        lh_pos = rdata.oMf[lh_pin_id].translation
        rf_pos = rdata.oMf[rf_pin_id].translation
        rh_pos = rdata.oMf[rh_pin_id].translation
        com = pinocchio.centerOfMass(rmodel, rdata, x[:dof])
        contact_poses += [np.concatenate([lf_pos, lh_pos, rf_pos,  rh_pos])]
        cs += [com]
        
    contact_poses = np.array(contact_poses)
    cs = np.array(cs)
    return cs, contact_poses

def get_phases_intervals_ddp(gait_sequence, N_ds=10, N_ss=30, N_start_ds = 20, N_end_ds = 20 ):
    phases_intervals = [N_start_ds]
    for name in gait_sequence[1:-1]:
        if name == 'DS':
            continue
        phases_intervals.append(N_ss)
        phases_intervals.append(N_ds)
    phases_intervals[-1] += (N_end_ds - N_ds)
    return phases_intervals
