import numpy as np
import pinocchio as pin


def get_gait(data):
    contact_activity = data['contact_activity'][()]
    lf = np.array(contact_activity['LF_ADAPTER_TO_FOOT']).flatten()
    rf = np.array(contact_activity['RF_ADAPTER_TO_FOOT']).flatten()
    lh = np.array(contact_activity['LH_ADAPTER_TO_FOOT']).flatten()
    rh = np.array(contact_activity['RH_ADAPTER_TO_FOOT']).flatten()
    phases = clean_phase(data['phases_intervals'][()])
    
    gait = []
    for phase in phases:
        idx_begin = phase[0]
        if lf[idx_begin] < 0.1:
            gait += ['LF']
        elif rf[idx_begin] < 0.1:
            gait += ['RF']
        elif lh[idx_begin] < 0.1:
            gait += ['LH']
        elif rh[idx_begin] < 0.1:
            gait += ['RH']
        else:
            gait += ['DS']
    return gait

def clean_phase(phases):
    if len(phases[-1]) == 0:
        return phases[:-1]
    else:
        return phases

def get_DT(phases):
    DT = []
    for i in range(len(phases)):
        DT += [phases[i][-1] - phases[i][0]]
    return np.array(DT)

def get_contact_sequence(gait, ee, phases):
    lf_traj = np.array(ee['LF_ADAPTER_TO_FOOT'])[:3].T
    rf_traj = np.array(ee['RF_ADAPTER_TO_FOOT'])[:3].T
    lh_traj = np.array(ee['LH_ADAPTER_TO_FOOT'])[:3].T
    rh_traj = np.array(ee['RH_ADAPTER_TO_FOOT'])[:3].T

    contact_sequence = []
    for i, gait_i in enumerate(gait):
        if gait_i != 'DS':
            continue
        t = phases[i][0]
        contact = dict()
        contact['LF_FOOT'] = lf_traj[t]
        contact['LH_FOOT'] = lh_traj[t]
        contact['RF_FOOT'] = rf_traj[t]
        contact['RH_FOOT'] = rh_traj[t]
        contact_sequence += [contact]
    return contact_sequence