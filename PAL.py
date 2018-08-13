import numpy as np
from scipy import stats
from numpy.linalg import det
import matplotlib.pyplot as plt
import pandas as pd
import time as tm


MAX_ITER = 100
ACTION_LABELS = ['n','s','e','w']
ACTION_COLOR = ['g','c','m','y']
ACTION_REAL_EFFECT = np.array([[0,1],[0,-1],[1,0],[-1,0]])
ACTIONS = np.arange(len(ACTION_LABELS))

INITIAL_STATE = None
STATES = None
GAMMA = None
EXPERIMENT_LABEL = None
WALLS = None
WORLD_SHAPE = None
CURRENT_LOCATION = None
CURRENT_STATE = None
MU = None
INITIALIZING_COV = None
COV = None
NEED_TO_PLAN = None
GID = None
NEW_STATE = False
NUMBER_OF_SAMPLES = None
WORLD_SAMPLE = None
DET_COV = None
INV_COV = None
NOISE_COV = None
DET_NOISE = None


def random_world(x,y,walls_config):
    global WALLS,WORLD_SHAPE
    WORLD_SHAPE = np.array([x,y])
    WALLS = [[[0,0],[x,0]],
              [[0,0],[0,y]],
              [[x,0],[x,y]],
              [[0,y],[x,y]]]
    for w in walls_config:
        for i in range(w[0]):
            h_or_v = np.zeros(2)
            h_or_v[np.random.randint(2)] = 1
            sx = np.random.randint(x+1-h_or_v[0]*w[1])
            sy = np.random.randint(y+1-h_or_v[1]*w[1])
            WALLS.append([[sx,sy],[sx,sy]+h_or_v*[w[1],w[1]]])
    WALLS = np.array(WALLS)




# random_world(5,5,((8,1),(3,2),(1,3)))
# np.save("world",WALLS)

def F(xy):
    result = np.array([stats.multivariate_normal.pdf(xy,MU[s],COV[s],allow_singular=True) for s in STATES])
    return result

def new_state():
    global STATES,GAMMA,NR_OF_STATES,\
        NEW_STATE,MU,COV,DET_COV,INV_COV
    NR_OF_STATES += 1
    NEW_STATE = True
    STATES = np.arange(NR_OF_STATES)
    GAMMA = np.concatenate([GAMMA,np.array([[STATES[-1]]]*len(ACTIONS))],axis=1)
    MU = np.concatenate([MU,[CURRENT_LOCATION]],axis=0)
    COV = np.concatenate([COV,[INITIALIZING_COV]],axis=0)
    DET_COV = np.concatenate([DET_COV,[np.linalg.det(INITIALIZING_COV)]], axis=0)
    INV_COV = np.concatenate([INV_COV,[np.linalg.inv(INITIALIZING_COV)]], axis=0)
    # print("adding the new state",STATES[-1],"centered in ",MU[STATES[-1]])
    return STATES[-1]

def distance_from_goal(s):
    return np.sum(np.square(MU[s]-GOAL),axis=-1)

def trajectory_dont_intersect_a_wall(source_loc,target_loc):
    for w in WALLS:
        if intersect(np.stack([source_loc,target_loc]), w):
            return False
    return True


def act(a):
    global CURRENT_LOCATION
    next_location = CURRENT_LOCATION + ACTION_REAL_EFFECT[a] + np.random.normal(0,NOISE,(1,2))[0]
    if trajectory_dont_intersect_a_wall(CURRENT_LOCATION,next_location):
       CURRENT_LOCATION = next_location
    return CURRENT_LOCATION

def plot_world():
    for w in WALLS:
        plt.plot(w[:, 0], w[:, 1],'k',linewidth=5)

def plot_pd():
    plt.clf()
    plt.scatter(MU[:, 0], MU[:, 1], marker="o", c="r")
    plt.scatter(CURRENT_LOCATION[0],CURRENT_LOCATION[1],marker="o",c='k',s=200)
    plt.scatter([GOAL[0]],
                [GOAL[1]], marker="o", c="r", s=400)
    if REACHED_GOAL:
        color = 'g'
    else:
        color = 'b'
    plt.scatter([MU[CURRENT_STATE][0]],
                [MU[CURRENT_STATE][1]], marker="o",c=color,s=400)
    for i in STATES:
        plt.annotate("s" + str(i), xy=MU[i] + 0.1)
    for w in WALLS:
        plt.plot(w[:, 0], w[:, 1], 'k',linewidth=5)
    for a in ACTIONS:
        for s in STATES:
            if np.sum(np.square(MU[GAMMA[a,s]]-MU[s])) > 1e-4:
                plt.arrow(*MU[s],*(MU[GAMMA[a,s]]-MU[s]),
                          head_width=.075,
                          length_includes_head=True,
                          color=ACTION_COLOR[a])
    plt.scatter(0,-1,marker="x",color="w")
    plt.scatter(WORLD_SAMPLE[:, 0], WORLD_SAMPLE[:, 1], marker=".",color="k",s=5)
    plt.annotate(EXPERIMENT_LABEL.replace("_",", ")+" working at goal %s"%GID,xy=np.array([0,-1]))

def plan():
    # print("planning")
    planned_actions = np.zeros_like(STATES)
    sorted_actions = np.argsort(distance_from_goal(GAMMA),axis=0)
    recent_states = O.tail(NR_OF_STATES)[O.columns[-1]].values.astype(np.int)
    for s in STATES:
        deadlock_in_s = True
        for a in sorted_actions[:,s]:
            if GAMMA[a,s] not in recent_states:
                planned_actions[s] = a
                deadlock_in_s = False
                break
        if deadlock_in_s:
            planned_actions[s] = np.random.choice(len(ACTIONS))
    return planned_actions


def intersect(AB,CD):
    return (np.sign(det(AB - CD[0])) == -np.sign(det(AB - CD[1])) and
            np.sign(det(CD - AB[0])) == -np.sign(det(CD - AB[1])))

def update_gamma(state,action):
    global GAMMA
    GAMMA[action][state] = np.argmax(np.array(
         [ALPHA*(s == GAMMA[action][state]).astype(np.float32) +\
            (1-ALPHA)*len(T.query("s == %i & a == %i & sp == %i"%(state,action,s))) for s in STATES]))

def update_f(state):
    global MU,COV,DET_COV,INV_COV
    obs_on_state =O.query("s==%s"%state)[['x','y']]
    MU[state] = BETA*MU[state]+(1-BETA)*obs_on_state.mean()
    if len(obs_on_state) > 1:
        COV[state] = BETA*COV[state]+(1-BETA)*np.maximum(obs_on_state.cov(),NOISE_COV)
    INV_COV[state] = np.linalg.inv(COV[state])
    DET_COV[state] = np.linalg.det(COV[state])

def check_reached_goal():
    result = np.sum(np.square(CURRENT_LOCATION-GOAL)) <= .5  and trajectory_dont_intersect_a_wall(CURRENT_LOCATION,GOAL)
    if result:
        print("goal %s achieved"%GID)
    return result

def PAL():
    global O,T,CURRENT_STATE,ITER, \
        CURRENT_LOCATION,REACHED_GOAL,COV,DET_COV,INV_COV
    ITER = 0
    while not REACHED_GOAL and ITER < MAX_ITER:
        pi = plan()
        need_to_plan = False
        while not need_to_plan:
            iter_after_last_plan = ITER
            act(pi[CURRENT_STATE])
            best_next_state = np.argmax(F(CURRENT_LOCATION))
            if F(CURRENT_LOCATION)[best_next_state] < EPSILON:
                best_next_state = new_state()
                need_to_plan = True
            elif best_next_state in O[iter_after_last_plan:]['s'].values:
                need_to_plan = True
            T.loc[len(T)] = [CURRENT_STATE,pi[CURRENT_STATE],best_next_state]
            O.loc[len(O)] = np.concatenate([CURRENT_LOCATION,[best_next_state]])
            update_gamma(CURRENT_STATE,pi[CURRENT_STATE])
            update_f(best_next_state)
            CURRENT_STATE = best_next_state
            REACHED_GOAL = check_reached_goal()
            if ITER%REPLANNING_FREQUENCY == 0:
                need_to_plan = True
            ITER += 1
            print(ITER)
    if ITER >= MAX_ITER and not REACHED_GOAL:
        print("no plan found for goal %s"%GID)
    logging()



def _coh(a,xy):
    s = np.argmax(F(xy))
    det_cov_a = DET_COV[GAMMA[a,s]]
    inv_cov_a = INV_COV[GAMMA[a,s]]
    det_cov_b = DET_NOISE
    cov_b = NOISE_COV
    mu_a = MU[GAMMA[a,s]]
    mu_b = xy
    if trajectory_dont_intersect_a_wall(xy,xy+ACTION_REAL_EFFECT[a]):
        mu_b += ACTION_REAL_EFFECT[a]
    d=2
    result = np.log(det_cov_a/det_cov_b)-d+np.trace(np.matmul(inv_cov_a,cov_b))+ \
           np.matmul((mu_a-mu_b).T,np.matmul(inv_cov_a,(mu_a-mu_b)))
    return result

def coherence():
    i,j = np.meshgrid(ACTIONS,np.arange(NUMBER_OF_SAMPLES))
    result =  np.array(list(map(lambda a,b:_coh(ACTIONS[a],WORLD_SAMPLE[b]),
        i.reshape(-1),
        j.reshape(-1)
        ))).mean()
    print(result)
    return result

def logging():
    global LOG,NEW_STATE
    LOG = LOG.append({
            'time': tm.time(),
            'goal_id':GID,
            'iter':ITER,
            'coherence':coherence(),
            'plan':NEED_TO_PLAN,
            'reached_goal':REACHED_GOAL,
            "nr_of_states":NR_OF_STATES,
            "new_state":NEW_STATE},ignore_index=True)
    NEW_STATE = False

def run_experiment(alpha, beta, epsilon, noise, replanning_frequency=10,number_of_goals=100):
    global ALPHA, BETA, NOISE, EPSILON, \
        REPLANNING_FREQUENCY,T,O, \
        CURRENT_STATE,REACHED_GOAL,GOAL, \
        LOG,GID,NEW_STATE,EXPERIMENT_LABEL, \
        GAMMA,NR_OF_STATES, INITIAL_STATE,\
        MU,CURRENT_LOCATION,COV,DET_COV,INV_COV,DET_NOISE,\
        STATES,NUMBER_OF_SAMPLES,WORLD_SAMPLE,INITIALIZING_COV,NOISE_COV
    EXPERIMENT_LABEL = 'a=%s_b=%s_e=%s_n=%s_p=%s_g=%s'%(alpha, beta, epsilon, noise, replanning_frequency,number_of_goals)
    NR_OF_STATES = 1
    STATES = np.arange(NR_OF_STATES)
    REACHED_GOAL = False
    STATES = np.arange(NR_OF_STATES)
    GAMMA = np.tile(STATES, (len(ACTIONS), 1))
    INITIAL_STATE = 0
    ALPHA = alpha
    BETA = beta
    NOISE = noise
    NOISE_COV = np.identity(2)*np.maximum(NOISE,1e-7)
    DET_NOISE = np.linalg.det(NOISE_COV)
    EPSILON = (1-epsilon) * stats.multivariate_normal.pdf([0, 0], [0, 0], INITIALIZING_COV)
    REPLANNING_FREQUENCY = replanning_frequency
    MU = np.array([np.random.uniform(size=(2,)) * WORLD_SHAPE for s in STATES])
    CURRENT_LOCATION = MU[INITIAL_STATE]
    INITIALIZING_COV = np.identity(2)
    COV = np.array([INITIALIZING_COV for _ in STATES])
    DET_COV = np.array([np.linalg.det(INITIALIZING_COV) for _ in STATES])
    INV_COV = np.array([np.linalg.inv(INITIALIZING_COV) for _ in STATES])
    CURRENT_STATE = INITIAL_STATE
    LOG = pd.DataFrame(columns=['goal_id','iter','coherence','plan','reached_goal',"nr_of_states",'new_state'])
    T = pd.DataFrame(columns=['s','a','sp'])
    O = pd.DataFrame([np.concatenate([CURRENT_LOCATION,[INITIAL_STATE]])],columns=['x','y','s'])
    WORLD_SAMPLE = []
    NUMBER_OF_SAMPLES = 100
    sample = np.copy(CURRENT_LOCATION)
    path_length = 300
    for _ in range(path_length):
        a = np.random.randint(0,len(ACTIONS))
        next_location = sample + ACTION_REAL_EFFECT[a] + np.random.normal(0, NOISE, (1, 2))[0]
        if trajectory_dont_intersect_a_wall(sample, next_location):
            sample = np.copy(next_location)
        WORLD_SAMPLE.append(np.copy(sample))
    WORLD_SAMPLE = np.array(WORLD_SAMPLE)[np.random.randint(0,path_length,NUMBER_OF_SAMPLES)]
    for GID in range(number_of_goals):
        GOAL = np.random.uniform(low=np.zeros((2,)),high=WORLD_SHAPE)
        print(GOAL)
        REACHED_GOAL = False
        plot_pd()
        plt.pause(.001)
        PAL()
    LOG.to_pickle(EXPERIMENT_LABEL)
    plot_pd()
    plt.pause(.001)
    plt.savefig(EXPERIMENT_LABEL + ".png")


def upload_world(world_file="world.npy"):
    global WALLS, WORLD_SHAPE
    WALLS = np.load(world_file)
    WORLD_SHAPE = WALLS[3, 1]

upload_world()

plt.clf()
for b in np.linspace(0.75,1,2):
    for e in np.linspace(0,1,5):
        for n in np.linspace(0,0.04,3):
            run_experiment(0.0,b,e,n,5,10)
