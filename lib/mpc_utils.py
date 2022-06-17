def mpcStep(mpc, n_step):
    xs, us, ffeas, xs_ref, us_ref= [], [], [], [], []
    for i in range(n_step):
        print(i)
        cur_x = mpc._solver.xs[0]
        us_f = mpc._solver.us
        xs_f = mpc._problem.rollout(us_f)
        x1 = xs_f[1]
        f = mpc.get_fopt(1) 
            
        mpc.step(x1[:rmodel.nq], x1[rmodel.nq:], f)
        mpc._solver.computeDynamicFeasibility()
        feas2 = mpc._solver.ffeas

        ffeas.append(feas2)
        xs.append(x1)
        us.append(us_f[0])
        xs_ref.append(np.array(mpc._solver.xs)) #solution of mpc at time step t
        us_ref.append(np.array(mpc._solver.us)) #solution of mpc at time step t
    
    return np.array(xs), np.array(us), np.array(ffeas), np.array(xs_ref),np.array(us_ref)

def mpcStepInit(mpc, mpc_init, n_step):
    xs, us, ffeas, xs_ref, us_ref= [], [], [], [], []
    for i in range(n_step):
        print(i)
        cur_x = mpc._solver.xs[0]
        us_f = mpc._solver.us
        xs_f = mpc._problem.rollout(us_f)
        x1 = xs_f[1]
        f = mpc.get_fopt(1) 
            
        #use standard mpc
        mpc_init.step(x1[:rmodel.nq], x1[rmodel.nq:], f)
        xs_init, us_init = mpc._solver.xs, mpc._solver.us

        mpc.step(x1[:rmodel.nq], x1[rmodel.nq:], f, xs_init, us_init)
        mpc._solver.computeDynamicFeasibility()
        feas2 = mpc._solver.ffeas

        ffeas.append(feas2)
        xs.append(x1)
        us.append(us_f[0])
        xs_ref.append(np.array(mpc._solver.xs)) #solution of mpc at time step t
        us_ref.append(np.array(mpc._solver.us)) #solution of mpc at time step t
    
    return np.array(xs), np.array(us), np.array(ffeas), np.array(xs_ref),np.array(us_ref)