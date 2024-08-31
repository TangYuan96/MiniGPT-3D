import os
import argparse
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

from pointllm.eval.evaluator import start_evaluation
from pointllm.data import ObjectPointCloudDataset

from minigpt4.common.eval_utils import prepare_texts, init_model
from minigpt4.conversation.conversation import CONV_VISION_minigptv2, CONV_VISION

conv_temp = CONV_VISION.copy()
conv_temp.system = ""

PROMPT_LISTS = [
    "What is this?",
    "This is an object of ",
    "Caption this 3D model in detail."
]


def load_dataset(data_path, anno_path, pointnum, conversation_types, use_color):
    print("Loading objaverse validation datasets.")
    dataset = ObjectPointCloudDataset(
        data_path=data_path,
        anno_path=anno_path,
        pointnum=pointnum,
        conversation_types=conversation_types,
        use_color=use_color,
        tokenizer=None  # * load point cloud only
    )
    return dataset


def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def start_generation(model, dataloader, annos, prompt_index, output_dir, output_file, args):
    qs = PROMPT_LISTS[prompt_index]

    results = {"prompt": qs}

    responses = []

    for batch in tqdm(dataloader):
        point_clouds = batch["point_clouds"].cuda()  # * tensor of B, N, C(3)
        object_ids = batch["object_ids"]  # * list of string

        texts = []
        texts.append(qs)
        texts = texts * point_clouds.size()[0]
        texts = prepare_texts(texts, conv_temp)

        model.eval()
        with torch.inference_mode():
            answers = model.generate(point_clouds, texts,
                                     num_beams=args.num_beams,
                                     max_new_tokens=args.max_new_tokens,
                                     min_length=args.min_length,
                                     top_p=args.top_p,
                                     repetition_penalty=args.repetition_penalty,
                                     length_penalty=args.length_penalty,
                                     temperature=args.temperature,
                                     do_sample=args.do_sample)

        outputs = []
        for answer in answers:
            answer = answer.lower().replace('<unk>', '').strip()
            answer = answer.split('###')[0]  # remove the stop sign '###'
            answer = answer.split('Assistant:')[-1].strip()
            outputs.append(answer)
 
        # saving results
        for obj_id, output in zip(object_ids, outputs):
            responses.append({
                "object_id": obj_id,
                "ground_truth": annos[obj_id],
                "model_output": output
            })

    results["results"] = responses

    os.makedirs(output_dir, exist_ok=True)
    # save the results to a JSON file
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)

    # * print info
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return results


def main(args):
    # * ouptut
    args.output_dir = os.path.join(args.out_path, "evaluation")

    # * output file 
    anno_file = os.path.splitext(os.path.basename(args.anno_path))[0]
    args.output_file = f"{anno_file}_Objaverse_{args.task_type}_prompt{args.prompt_index}.json"
    args.output_file_path = os.path.join(args.output_dir, args.output_file)

    # * First inferencing, then evaluate
    if not os.path.exists(args.output_file_path):
        # * need inferencing
        # * load annotation files
        with open(args.anno_path, 'r') as fp:
            annos = json.load(fp)

        dataset = load_dataset(args.data_path, args.anno_path, args.pointnum, ("simple_description",), args.use_color)
        dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers)


        model = init_model(args)
        model=model.eval()

        conv_temp = CONV_VISION.copy()
        conv_temp.system = ""


        annos = {anno["object_id"]: anno["conversations"][1]['value'] for anno in annos}

        print(f'[INFO] Start generating results for {args.output_file}.')
        results = start_generation(model, dataloader, annos, args.prompt_index, args.output_dir, args.output_file, args)

        # * release model and release cuda memory
        del model

        torch.cuda.empty_cache()
    else:
        # * directly load the results
        print(f'[INFO] {args.output_file_path} already exists, directly loading...')
        with open(args.output_file_path, 'r') as fp:
            results = json.load(fp)

    if args.start_eval:
        evaluated_output_file = args.output_file.replace(".json", f"_evaluated_{args.gpt_type}.json")
        eval_type_mapping = {
            "captioning": "object-captioning",
            "classification": "open-free-form-classification"
        }
        start_evaluation(results, output_dir=args.output_dir, output_file=evaluated_output_file,
                         eval_type=eval_type_mapping[args.task_type], model_type=args.gpt_type, parallel=True,
                         num_workers=20)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, default="./output/test")

    # * dataset type
    parser.add_argument("--data_path", type=str, default="./data/objaverse_data", required=False)
    parser.add_argument("--anno_path", type=str,
                        default="./data/anno_data/PointLLM_brief_description_val_200_GT.json",
                        required=False)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--use_color", action="store_true", default=True)

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=15)  # 32
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)

    # * evaluation setting
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--start_eval", action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-4-0613",
                        choices=["gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview"],
                        help="Type of the model used to evaluate.")
    parser.add_argument("--task_type", type=str, default="classification", choices=["captioning", "classification"],
                        help="Type of the task to evaluate.")


    parser.add_argument("--cfg-path", default='./eval_configs/paper_result/benchmark_evaluation_paper.yaml',
                        help="path to configuration file.")

    ##  control the generation
    parser.add_argument("--max_new_tokens", type=int, default=150, help="max number of generated tokens")
    parser.add_argument("--min_length", type=int, default=10, help="min number of generated tokens")
    parser.add_argument("--num_beams", type=int, default=2, help="the bigger, the more accurater result")
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--do_sample", type=bool, default=True)


    parser.add_argument("--options", nargs="+",
                        help="override some settings in the used config, the key-value pair "
                             "in xxx=yyy format will be merged into config file (deprecate), "
                             "change to --cfg-options instead.",
                        )


    args = parser.parse_args()

    # * check prompt index
    # * * classification: 0, 1 and captioning: 2. Raise Warning otherwise.
    if args.task_type == "classification":
        if args.prompt_index != 0 and args.prompt_index != 1:
            print("[Warning] For classification task, prompt_index should be 0 or 1.")
    elif args.task_type == "captioning":
        if args.prompt_index != 2:
            print("[Warning] For captioning task, prompt_index should be 2.")
    else:
        raise NotImplementedError

    main(args)
