import torch
from stable_baselines3 import PPO

# The official, robust pattern for exporting SB3 policies to ONNX
class OnnxablePolicy(torch.nn.Module):
    def __init__(self, mlp_extractor, action_net):
        super().__init__()
        # Isolate standard child modules directly
        self.mlp_extractor = mlp_extractor
        self.action_net = action_net

    def forward(self, observation):
        # mlp_extractor maps inputs to a tuple: (latent_policy, latent_value)
        latent_pi, _ = self.mlp_extractor(observation)
        
        # Pass the isolated policy features directly into the action weights
        return self.action_net(latent_pi)

def main():
    print("Loading Stable-Baselines3 model...")
    # Load the zip to the CPU for clean compilation
    model = PPO.load("D:\\Python_Engine_2\\ppo_local_planner_v1.zip", device="cpu")
    
    # Extract the core submodules required for local planner inference
    onnxable_model = OnnxablePolicy(
        model.policy.mlp_extractor, 
        model.policy.action_net
    )
    onnxable_model.eval() # Freeze layers

    # Map the tracer matrix to your exact observation space length (26 inputs)
    dummy_input = torch.zeros((1, 26), dtype=torch.float32)

    onnx_filename = "robot_local_planner.onnx"
    print(f"Compiling computational graph to {onnx_filename}...")

    # Export cleanly using basic layer-to-layer tracing
    torch.onnx.export(
        onnxable_model,
        dummy_input,
        onnx_filename,
        export_params=True,             # Store weights in the binary
        opset_version=12,               # Highly compatible execution opset
        input_names=['observation'],    # C++ graph input gateway
        output_names=['actions'],       # C++ graph output velocity extraction
        dynamic_axes={'observation': {0: 'batch_size'}, 'actions': {0: 'batch_size'}}
    )

    print("ONNX export complete! Your deployment brain is ready.")

if __name__ == "__main__":
    main()