# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import argparse
import os
import time
from pathlib import Path

import onnxruntime_genai as og

def run(args: argparse.Namespace):
    print("Loading model...")
    config = og.Config(args.model_path)
    config.clear_providers()
    if args.execution_provider != "cpu":
        print(f"Setting model to {args.execution_provider}...")
        config.append_provider(args.execution_provider)
    model = og.Model(config)
    print("Model loaded")

    processor = model.create_multimodal_processor()
    tokenizer_stream = processor.create_stream()

    interactive = not args.non_interactive
    project_dir = Path(__file__).parent.absolute()  # Project folder: /home/artem/Documents/phi-multimodal/

    while True:
        # Use img.jpg from the project folder by default
        default_image_path = project_dir / "img.jpg"
        image_paths = []

        if interactive:
            try:
                import readline
                readline.set_completer_delims(" \t\n;")
                readline.parse_and_bind("tab: complete")
                # Simple completion could be added here if needed
            except ImportError:
                pass
            user_input = input("Image Path (comma separated; leave empty for img.jpg): ").strip()
            image_paths = [path.strip() for path in user_input.split(",")] if user_input else [str(default_image_path)]
        else:
            image_paths = args.image_paths if args.image_paths else [str(default_image_path)]

        image_paths = [path for path in image_paths if path]  # Filter out empty paths

        images = None
        prompt = "<|user|>\n"

        # Process images
        if not image_paths:
            print("No image provided")
        else:
            for i, image_path in enumerate(image_paths):
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                print(f"Using image: {image_path}")
                prompt += f"<|image_{i+1}|>\n"
            images = og.Images.open(*image_paths)

        # Get prompt
        if interactive:
            text = input("Prompt: ")
        else:
            text = args.prompt if args.prompt else "Describe the image."
        prompt += f"{text}<|end|>\n<|assistant|>\n"

        print("Processing inputs...")
        inputs = processor(prompt, images=images)
        print("Processor complete.")

        print("Generating response...")
        params = og.GeneratorParams(model)
        params.set_inputs(inputs)
        params.set_search_options(max_length=7680)

        generator = og.Generator(model, params)
        start_time = time.time()

        while not generator.is_done():
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end="", flush=True)

        print()
        total_run_time = time.time() - start_time
        print(f"Total Time : {total_run_time:.2f}")

        for _ in range(3):
            print()

        del generator  # Free resources

        if not interactive:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_path", type=str, required=True, help="Path to the folder containing the model"
    )
    parser.add_argument(
        "-e", "--execution_provider", type=str, required=True, choices=["cpu", "cuda", "dml"], help="Execution provider to run model"
    )
    parser.add_argument(
        "--image_paths", nargs='*', type=str, required=False, help="Path to the images, mainly for CI usage"
    )
    parser.add_argument(
        '-pr', '--prompt', required=False, help='Input prompts to generate tokens from, mainly for CI usage'
    )
    parser.add_argument(
        '--non-interactive', action=argparse.BooleanOptionalAction, required=False, help='Non-interactive mode, mainly for CI usage'
    )
    args = parser.parse_args()
    run(args)