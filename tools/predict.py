import sys
sys.path.append("/home/viktor/Documents/kaggle/kaggle_llm/src")
from kaggle_llm.adapted_models import *
from kaggle_llm.core import (
    ROOT_PATH,
    DataCollatorForMultipleChoice,
    DataCollatorForMultipleChoicePrompting,
    WORK_DIRS_PATH,
    infer_pred_from_scores,
    get_tokenize_dataset_from_df,
    get_mcp_tokenize_dataset_from_df,
    drop_df_cols_for_dataset,
    WrappedPeftModel,
    add_context
)
from transformers import AutoModelForMultipleChoice, Trainer, AutoTokenizer, TrainingArguments
from peft import PeftModel
from sklearn.preprocessing import normalize
from pathlib import Path
import pandas as pd
import argparse
import yaml
import json
from memory_profiler import profile

# @profile
def main(
        input_df_path: str,
        output_dir: str,
        base_models_dir: str = "",
        device: str = "cuda",
        max_input: int = 1000000,
        preprocess_type: str = "deotte",
):
    input_df_path = Path(input_df_path)
    assert input_df_path.is_file(), f"{input_df_path} not found"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(ROOT_PATH / "configs" / "submission.yaml", "rb") as f:
        submission_config = yaml.load(f, yaml.FullLoader)

    print(json.dumps(submission_config, indent=4))
    models = submission_config["models"]
    configs = submission_config["configs"]
    df = pd.read_csv(input_df_path)
    df = drop_df_cols_for_dataset(df)
    
    if "add_context" in submission_config and submission_config["add_context"]:
        df = add_context(df)

    for i, (load_from, config_file) in enumerate(zip(models, configs)):
        print(f"initting models [{i}]")
        abs_load_from = WORK_DIRS_PATH / load_from
        abs_config = WORK_DIRS_PATH / config_file
        is_peft = "peft" in abs_load_from.parent.name
        tokenizer = AutoTokenizer.from_pretrained(abs_load_from)
        if is_peft:
            config = json.loads(Path(abs_load_from / "adapter_config.json").read_text())
            base_model_path = Path(config["base_model_name_or_path"])
            print(f"{base_models_dir = }")
            if base_models_dir != "":
                base_model_path = Path(base_models_dir) / base_model_path.name
                print(f"base_model_dir given, overriding peft base_model_path to: {base_model_path}")
            model = AutoModelForMultipleChoice.from_pretrained(base_model_path, load_in_8bit=True, device_map='auto')
            model = WrappedPeftModel.from_pretrained(model, abs_load_from)
            if hasattr(model.base_model, "load_extra_modules"):
                model.base_model.load_extra_modules(abs_load_from)
            kwargs = dict(
                remove_unused_columns=False,
                label_names=["labels"],
            )
        else:
            model = AutoModelForMultipleChoice.from_pretrained(abs_load_from)
            kwargs = {}
        print(f"initted models [{i}]")

        print(f"initting tokenizer and trainer [{i}]")
        print(submission_config)
        
        if 'separate_prompt_and_context' in submission_config and submission_config['separate_prompt_and_context']:
            def get_context(text):
                x = text.split(" ### ")
                # remove empty strings
                x = [x for x in x if len(x) > 0]
                
                assert len(x) == 2, f"Unsuccesful context splitting . len(x) = {len(x)}, x={x}"
                return x[0]
            
            
            df['context'] = df['prompt'].apply(lambda x: get_context(x))
            
            def get_prompt(text):
                x = text.split(" ### ")
                # remove empty strings
                x = [x for x in x if len(x) > 0]
                assert len(x) == 2, f"Unsuccesful prompt splitting . len(x) = {len(x)}"
                return x[1]
            
            df['prompt'] = df['prompt'].apply(lambda x: get_prompt(x))
        
            # df['context_sentences'] = df['context'].apply(lambda x: x.split(". "))
            # df = df.explode('context_sentences').reset_index(drop=True)
            # df['context'] = df['context_sentences']
            
            
            # give answer as context 
            # df['context'] = df.apply(lambda x: x[x['answer']], axis=1)
            
            # df['context'] = df.apply(lambda x: x['context'][:900] + " " + x[x['answer']], axis=1)
            
            # df_1 = df.copy()
            # df_1['context'] = df_1.apply(lambda x: x['A'], axis=1)
            
            # df_2 = df.copy()
            # df_2['context'] = df_2.apply(lambda x: x['B'], axis=1)
            
            # df_3 = df.copy()
            # df_3['context'] = df_3.apply(lambda x: x['C'], axis=1)
            
            # df_4 = df.copy()
            # df_4['context'] = df_4.apply(lambda x: x['D'], axis=1)
            
            # df_5 = df.copy()
            # df_5['context'] = df_5.apply(lambda x: x['E'], axis=1)
            
            # df = pd.concat([df_1, df_2, df_3, df_4, df_5]).reset_index(drop=True)
            
            
        
        
        if submission_config['tokenization'] == 'get_tokenize_dataset_from_df':
            tokenized_dataset = get_tokenize_dataset_from_df(df, tokenizer, preprocess_type, max_input)
        elif submission_config['tokenization'] == 'get_mcp_tokenize_dataset_from_df':
            tokenized_dataset = get_mcp_tokenize_dataset_from_df(df, tokenizer)
        training_args = TrainingArguments(
            per_device_eval_batch_size=1,
            output_dir="/tmp/kaggle_llm_pred",
            **kwargs,
            fp16=True if device == "cuda" else False,
        )
        
        if submission_config['data_collator'] == "DataCollatorForMultipleChoice":
            data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
        elif submission_config['data_collator'] == "DataCollatorForMultipleChoicePrompting":
            data_collator = DataCollatorForMultipleChoicePrompting(tokenizer=tokenizer)
        
        trainer = Trainer(
            model=model.eval(),
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=None,
        )
        print(f"initted tokenizer and trainer [{i}]")

        print(f"predicting [{i}]")
        with torch.no_grad():
            preds = trainer.predict(tokenized_dataset)
        print(f"predicted [{i}]")

        model_output_path = output_dir / (load_from.replace("/", "-") + ".csv")
        normalised_preds = normalize(-preds.predictions)
        preds_df = pd.DataFrame({
            "id": df.index,
            **{f"score{k}": normalised_preds[:, k] for k in range(5)}
        }).set_index("id")
        preds_df = infer_pred_from_scores(preds_df)
        preds_df.to_csv(model_output_path, header=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("df_path")
    parser.add_argument("--output-dir", default=f"{str(ROOT_PATH)}/preds")
    parser.add_argument("--base-models-dir", default="")
    parser.add_argument("--device", default="cuda", type=str, help="cuda or cpu", required=False)
    parser.add_argument("--max-input", default="cuda", type=int, help="number of inputs", required=False)
    parser.add_argument("--preprocess_type", default="cuda", type=str, help="preprocess_type (deotte or viktor/sumo)", required=False)
    args, _ = parser.parse_known_args()
    main(args.df_path, args.output_dir, args.base_models_dir, args.device, args.max_input, args.preprocess_type)
