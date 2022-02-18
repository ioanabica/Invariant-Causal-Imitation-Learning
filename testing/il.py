from __head__ import *

from agent import OAStableAgent
from contrib.energy_model import EnergyModel
from contrib.env_wrapper import EnvWrapper, get_test_mult_factors
from network import (
    StudentNetwork, FeaturesEncoder, FeaturesDecoder, ObservationsDecoder, EnvDiscriminator, MineNetwork)
from student import (
    BaseStudent,
    ICILStudent,
)
from testing.train_utils import fill_buffer, save_results, make_agent


def make_student(run_seed, config):
    env = gym.make(config['ENV'])
    trajs_path = get_trajs_path(config['ENV'], 'student_' + config['ALG'], env_id='student', run_seed=run_seed)
    model_path = get_model_path(config['ENV'], 'student_' + config['ALG'], run_seed=run_seed)

    state_dim = env.observation_space.shape[0] + config['NOISE_DIM']
    action_dim = env.action_space.n
    num_training_envs = config['NUM_TRAINING_ENVS']

    run_seed = run_seed
    batch_size = config['BATCH_SIZE']
    teacher = make_agent(config['ENV'], config['EXPERT_ALG'], config['NUM_TRAINING_ENVS']);
    teacher.load_pretrained()

    buffer = fill_buffer(trajs_path=teacher.trajs_paths, batch_size=batch_size, run_seed=run_seed,
                         traj_shift=config['TRAJ_SHIFT'], buffer_size_in_trajs=config['NUM_TRAJS_GIVEN'],
                         sampling_rate=config['SAMPLING_RATE'])

    if (buffer.total_size < batch_size):
        batch_size = buffer.total_size

    energy_model = EnergyModel(in_dim=state_dim,
                               width=config['MLP_WIDTHS'],
                               batch_size=batch_size,
                               adam_alpha=config['ADAM_ALPHA'],
                               buffer=buffer,
                               sgld_buffer_size=config['SGLD_BUFFER_SIZE'],
                               sgld_learn_rate=config['SGLD_LEARN_RATE'],
                               sgld_noise_coef=config['SGLD_NOISE_COEF'],
                               sgld_num_steps=config['SGLD_NUM_STEPS'],
                               sgld_reinit_freq=config['SGLD_REINIT_FREQ'],
                               )
    energy_model.train(num_updates=config['NUM_STEPS_TRAIN_ENERGY_MODEL'])

    causal_features_encoder = FeaturesEncoder(input_size=state_dim,
                                              representation_size=config['REP_SIZE'],
                                              width=config['MLP_WIDTHS'])

    causal_features_decoder = FeaturesDecoder(action_size=action_dim,
                                              representation_size=config['REP_SIZE'],
                                              width=config['MLP_WIDTHS'])

    observations_decoder = ObservationsDecoder(representation_size=config['REP_SIZE'], out_size=state_dim,
                                               width=config['MLP_WIDTHS'])

    policy_network = StudentNetwork(in_dim=config['REP_SIZE'], out_dim=action_dim,
                                    width=config['MLP_WIDTHS'])

    env_discriminator = EnvDiscriminator(representation_size=config['REP_SIZE'],
                                         num_envs=config['NUM_TRAINING_ENVS'], width=config['MLP_WIDTHS'])

    noise_features_encoders = [FeaturesEncoder(input_size=state_dim,
                                               representation_size=config['REP_SIZE'],
                                               width=config['MLP_WIDTHS']) for i in range(num_training_envs)]
    noise_features_decoders = [FeaturesDecoder(action_size=action_dim,
                                               representation_size=config['REP_SIZE'],
                                               width=config['MLP_WIDTHS']) for i in range(num_training_envs)]

    mine_network = MineNetwork(x_dim=config['REP_SIZE'], z_dim=config['REP_SIZE'], width=config['MLP_WIDTHS'])

    return ICILStudent(
        env=env,
        trajs_paths=trajs_path,
        model_path=model_path,
        num_training_envs=num_training_envs,
        teacher=teacher,
        causal_features_encoder=causal_features_encoder,
        noise_features_encoders=noise_features_encoders,
        causal_features_decoder=causal_features_decoder,
        noise_features_decoders=noise_features_decoders,
        observations_decoder=observations_decoder,
        env_discriminator=env_discriminator,
        policy_network=policy_network,
        energy_model=energy_model,
        mine_network=mine_network,
        buffer=buffer,
        adam_alpha=config['ADAM_ALPHA'],
    )


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default='CartPole-v1')
    parser.add_argument("--num_trajectories", default=20, type=int)
    parser.add_argument("--trial", default=0, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = init_arg()

    config = {
        'ENV': args.env_name,
        'ALG': 'ICILStudent',
        'NUM_TRAJS_GIVEN': args.num_trajectories,
        'NUM_TRAINING_ENVS': 2,
        'NOISE_DIM': 4,

        'REP_SIZE': 16,

        'TRAJ_SHIFT': 20,
        'SAMPLING_RATE': 5,

        'NUM_STEPS_TRAIN': 10000,
        'NUM_TRAJS_VALID': 100,
        'NUM_REPETITIONS': 10,

        'BATCH_SIZE': 64,
        'MLP_WIDTHS': 64,
        'ADAM_ALPHA': 1e-3,

        'SGLD_BUFFER_SIZE': 10000,
        'SGLD_LEARN_RATE': 0.01,
        'SGLD_NOISE_COEF': 0.01,
        'SGLD_NUM_STEPS': 100,
        'SGLD_REINIT_FREQ': 0.05,
        'NUM_STEPS_TRAIN_ENERGY_MODEL': 1000,
    }

    config['EXPERT_ALG'] = yaml.load(open('testing/config.yml'), Loader=yaml.FullLoader)[config['ENV']]
    print("Config: %s" % config)

    TRIAL = args.trial
    print("Trial number %s" % TRIAL)

    results_dir_base = 'testing/results/'
    results_dir = os.path.join(results_dir_base, config['ENV'], str(config['NUM_TRAJS_GIVEN']), config['ALG'])

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    config_file = 'trial_' + str(TRIAL) + '_' + 'config.pkl'

    results_file_name = 'trial_' + str(TRIAL) + '_' + 'results.csv'
    results_file_path = os.path.join(results_dir, results_file_name)

    if os.path.exists(os.path.join(results_dir, config_file)):
        raise NameError('CONFIG file already exists %s. Choose a different trial number.' % config_file)
    pickle.dump(config, open(os.path.join(results_dir, config_file), 'wb'))

    for run_seed in range(config['NUM_REPETITIONS']):
        print("Run %s out of %s" % (run_seed + 1, config['NUM_REPETITIONS']))
        student = make_student(run_seed, config)
        student.train(num_updates=config['NUM_STEPS_TRAIN'])

        env_wrapper_out_of_sample = EnvWrapper(env=gym.make(config['ENV']), mult_factor=get_test_mult_factors(3), idx=3,
                                               seed=1)
        action_match, return_mean, return_std = student.test(
            num_episodes=config['NUM_TRAJS_VALID'], env_wrapper=env_wrapper_out_of_sample)

        result = (action_match, return_mean, return_std)
        print("Reward for test environment for run %s: %s" % (run_seed + 1, return_mean))
        save_results(results_file_path, run_seed, action_match, return_mean, return_std)

    results_trial = pd.read_csv(
        'testing/results/' + config['ENV'] + '/' + str(config['NUM_TRAJS_GIVEN']) + '/' + config['ALG'] + '/trial_' +
        str(TRIAL) + '_results.csv', header=None)

    print("Average reward for 10 repetitions: %s" % np.mean(results_trial[2].values))
