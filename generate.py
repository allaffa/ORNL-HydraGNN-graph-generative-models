import os, yaml, argparse
import torch

import hydragnn
from hydragnn.utils.distributed import get_distributed_model


from src.utils import data_utils as du
#from src.processes.marginal_diffusion import MarginalDiffusionProcess
from src.processes.marginal_diffusion import EquivariantDiffusionProcess
from src.utils.train_utils import insert_t, postprocess_model_outputs


def pred_fn(model, data, dp):
    data_tx, _ = insert_t(data, data.t, dp.timesteps)
    model.eval()
    out = postprocess_model_outputs(model(data_tx), data)
    atom_ident_noise, atom_pos_noise = out[0], out[1]
    noise_data = data.clone().detach_()
    noise_data.x = atom_ident_noise
    noise_data.pos = atom_pos_noise
    return noise_data


def load_model(args):
    # load the config
    with open(os.path.join(args.run_name, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    verbosity = config["Verbosity"]["value"]["level"]
    # Create the model from the config specifications
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"]["value"],
        verbosity=verbosity,
    )
    # Distribute the model across ranks (if necessary).
    # We need to unravel the model dictionary from the DDP wrapper before loading it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(
        os.path.join(args.run_name, "checkpoints/model_best.pt"),
        map_location=device
    )["model_state_dict"]
    # Remove 'module.' prefix
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return config, model


def generate(args):
    """
    Generates molecular structures using a diffusion process and a pretrained model.

    args:
        args (argparse.Namespace): Command-line arguments containing the following attributes:
            - data_path (str): Path to the data used to define the marginal distribution.
            - diffusion_steps (int): Number of diffusion steps to use in the generative process.
            - num_gen (int): Number of molecular structures to generate.
            - model (str): Path or identifier for loading the pretrained model.
    """

    # load model, get device
    _, model = load_model(args)
    device = hydragnn.utils.distributed.get_device()

    # setup diffusion process
    """
    dp = MarginalDiffusionProcess(
        args.diffusion_steps, marg_dist=du.get_marg_dist(root_path=args.data_path)
    )
    """
    dp = EquivariantDiffusionProcess(args.diffusion_steps)

    # Define prior distribution for the generative model
    prior_dist_state = dp.prior_dist(torch.randint(5, 20, (args.num_gen,)), 5, 3).to(
        device
    )

    # Sample from the prior distribution
    prior_samples = dp.sample_from_dist(prior_dist_state)

    # Denoise samples and generate data
    predictor = lambda data: pred_fn(model, data, dp)
    gen_data = dp.reverse_process_sample(prior_samples, predictor)

    gen_data_list = gen_data.to_data_list()
    # Write PDB files for generated data
    for i, gd in enumerate(gen_data_list):
        # postprocess by subtracting off CoM
        gd.pos = gd.pos - gd.pos.mean(dim=0, keepdim=True)
        out_path = os.path.join("models", "test", "structures", f"gen_{i}.pdb")
        du.write_pdb_file(gd, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Create default log name if not specified.
    parser.add_argument("-g", "--num_gen", type=int, default=100)
    parser.add_argument(
        "-d", "--data_path", type=str, default="examples/qm9/dataset"
    )
    parser.add_argument("-ds", "--diffusion_steps", type=int, default=100)
    parser.add_argument("-l", "--run_name", type=str, default="test")
    # parser.add_argument("-m", "--model", type=str, help='path to the trained model' )

    # Store the arguments in args.
    args = parser.parse_args()
    generate(args)
