from llava.serve.cli_pe import main, load_pretrained_model, get_model_name_from_path
import argparse
import pathlib


def load_model_for_session():
    args = args_with_image()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)
    return (tokenizer, model, image_processor, context_len)


# Write a function take in the path of image file, and return args with that image file.
def args_with_image(path: str = ''):
    # Create args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/LLaVA-Lightning-MPT-7B-preview")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    # Add image file to args
    args.image_file = path
    return args

def describe(path: str, prompt: str, model):
    args = args_with_image(path)
    answer = main(args, prompt, model)
    return answer

# Describe twice
def describe_twice(path: str):
    args = args_with_image(path)
    answer = main(args, 'describe the image')
    new_prompt = str(answer) + ' describe the mental state of the subject'
    new_answer = main(args, new_prompt)
    return new_answer

if __name__ == "__main__":
    prompt = 'describe mental state of the subject in the image'
    image_list = [str(file) for file in pathlib.Path('./image_samples').rglob("*") if file.is_file()]
    model = load_model_for_session() # load model only once during session to reduce time running main()
    print()
    for image in image_list:
        answer = describe(image, prompt, model)
        print('Describing picture: ' + image)
        print(answer)
        print()

    # args = args_with_image('./images/xiaoman.jpg')
    # answer = main(args, prompt)
    # print(answer)
    #main(args)
