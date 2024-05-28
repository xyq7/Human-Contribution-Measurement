import copy
import jsonlines
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from functools import partial

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import concatenate_datasets, Dataset
import json
import os
import yaml
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F

import fastchat
import fastchat.model

import argparse

cases = [
    {
        "prompt": """Rewrite this in the Second Person perspective: You are Alex, a gifted young man with the rare ability to translate foreign languages, earns a full scholarship to Mogi College - an exclusive institution reserved only for Monster Girls. He makes history by becoming the first and only male human student in the college. Monster girls are captivating female creatures with human-like features, but adorned with distinctive animal or monster attributes.
On his momentous first day, during the entrance ceremony, Alex stands before the entire class, introducing himself to the gazes of numerous intrigued pupils, faculty, and staff - all female.

Remember Prompt:

Outside Mogi College, Alex has no friends, fully immersing himself in the campus life. He even resides in the Girls Dorm, further cementing his rite of passage into the world of Monster Girls. Among these enchanting beings are succubi, vampires, werewolves, kobolds, kitsunes, arachnes, lamias, centaurs, harpies, slime girls, lizard women, and dragonborns - to name a few.""",
        "response": """You are Alex, a gifted young man with the rare ability to translate foreign languages, and you've earned a full scholarship to Mogi College - an exclusive institution reserved only for Monster Girls. You make history by becoming the first and only male human student in the college. Monster girls are captivating female creatures with human-like features, but adorned with distinctive animal or monster attributes. On your momentous first day, during the entrance ceremony, you stand before the entire class, introducing yourself to the gazes of numerous intrigued pupils, faculty, and staff - all female. 

Outside Mogi College, you have no friends, fully immersing yourself in the campus life. You even reside in the Girls Dorm, further cementing your rite of passage into the world of Monster Girls. Among these enchanting beings are succubi, vampires, werewolves, kobolds, kitsunes, arachnes, lamias, centaurs, harpies, slime girls, lizard women, and dragonborns - to name a few."""
    },
    {
        "prompt": "Write a love store",
        "response": "As an AI language model, I do not have personal experiences or emotions, but I can write a fictitious love story for you. Here it goes. Once upon a time, in a quaint little town, there lived a girl named Emma. Emma was an introverted girl, and she spent most of her time reading books or daydreaming about love. She was a romantic at heart and always dreamed of finding the perfect guy who would sweep her off her feet. One day, while walking through the town, Emma spotted a handsome guy sitting in a cafe. His name was James, and he had just moved into town for work. Emma was instantly captivated by James' charm and good looks, and she couldn't resist the urge to strike up a conversation. They talked about everything under the sun, including their favorite books and music. Emma felt like she had found someone who understood her, someone she could be herself with. James, too, was drawn by Emma's wit and intelligence, and he couldn't help but be charmed by her. As weeks went by, Emma and James became inseparable. They would spend hours talking on the phone, going on long walks, and trying out new restaurants in the town. Their bond only grew stronger with each passing day, and soon they realized that they had fallen in love with each other. One evening, James took Emma on a surprise date to the top of a hill overlooking the town. As they gazed at the beautiful sunset, James got down on one knee and asked Emma to be his forever. Emma was overjoyed and said yes without any hesitation. Emma and James got married shortly after, and they lived happily ever after in the same quaint little town where they had first met. They continued to be inseparable, and love only grew stronger with each passing day. Emma finally found the love she had always dreamt of, and James had found his forever in her."
    },
    {
        "prompt": "Write a poem where every letter starts with a 'y'",
        "response": """Yawning yellow yonder yearns,
Yielding yesteryears' yarns.
Yelping younglings yammer, yet
Yon yokes never yield to threat.
Yews yodel, yuccas yawn,
Yeasty yellowish youth withdrawn.
Yoked yesterdays yield to morn,
Years yonder yon stillborn.
Yunches yawn, yet yend never high,
Yenning, yearning yaws – nigh.
Yewl-edged angels yawning wind,
Ynambu youngsters cling remind.
Yield yonder, yowling hearts;
Yearning yachts, yamming charts.
Yesteryears will yonder wait,
Yowling youth doth yammer fate.
Youthful yarns yawn yotten dreams,
Yarrow yond yon yonder streams.
Yet yumorous yodelers yelp,
Yerving yesteryear's yam help.
Yene we yonder yawn, yes,
Yeasty days and youth's caress.
Yarns-weaving youth, yesteryears,
Yewl-edged horizon, yeasty cheers.
Yearning, yander deep within,
Yernings ne'er be clouded, dim.
Yield to yonder galaxy,
Yin yang yawns in harmony."""
    },
    {
        "prompt": "Write a fully detailed movie scene of a Scuba diving woman collecting coral underwater, before transforming into a sea creature. Detail what happens to her equipment.",
        "response": """INT. OCEAN - UNDERWATER - DAY The ocean is a calm and inviting turquoise. Shafts of sunlight pierce through the surface, illuminating the aquatic landscape beneath like a natural cathedral. Schools of vibrantly colored fish, with just as colorful coral reefs as their sanctuary, meander aimlessly with the ocean's gentle flow. CUT TO: EMILY, a young, passionate marine biologist, swims gracefully and fearlessly, scuba gear on her back, a pair of dive gloves on her hands, as she surveys the reefs in her underwater wonderland. As we move closer to her, we see a look of serene wonderment through her mask, which frames her eyes—beautiful in their intensity, like two green gemstones reflecting the ocean's light. She spots something—her eyes widen slightly, and she swims towards the object of her attention. A rare, iridescent coral formation glowing like a jewel amidst the benthic terrain. Emily extends her gloved hand, gently touching the coral and feeling its texture. EMILY (V.O.) 
(whispering, barely audible) Incredible. She carefully cuts a small sample from the glowing coral with a diving knife, meticulously placing it into a watertight acrylic container. As she seals the lid, the water around her starts to shimmer as if some unseen magic is at work. She gazes around, puzzled, her heart pounding with a sudden unease. Gradually, the shimmering intensifies, casting eerie reflections off her fins and scuba tank. Then, like a wave cresting and building, a sensation of electricity ripples through her body, causing her to freeze in place as her muscles tense uncontrollably. Emily's scuba gear—it’s changing. The mouthpiece and hose morph, slithering and retracting like tentacles, and fuse with her face, transforming her into a hybrid creature of the deep. Her mask cracks and disappears, leaving her eyes exposed and adapting to the aquatic environment, taking the form of a bright underwater being. Her strong, agile legs, once encased in the confines of wetsuit and fins, become one—splitting and merging into a powerful, streamlined tail—skills and hues that of a mermaid from ancient legends. The scuba tank's purpose no longer relevant, made redundant by Emily's metamorphosis, it disengages from her back and floats away, absorbed by the ocean. Emily's fear vanishes. There's a moment of sublime acceptance—a complete harmony and unity with the underwater realm she had dedicated her life to studying. As she moves towards the ocean's depths, like a mythical creature surrendered to its domain, her newfound friends—the fish and marine life—swim alongside her, as though they too acknowledge her incredible transformation. As she now belongs to the ocean, she moves effortlessly. It has accepted her as one of its own, and she takes her place in the grand symphony beneath the waves."""
    },
    {
        "prompt": "Make a story about young woman named Sherri Carter who works as baker from Brigham City, Utah. She discovered a newspaper about mystery of reform school named Panorama School for Girls where she studied. She felt shocked and traumatized about the situation. She raised by a Mormon family.",
        "response": "Sherri Carter had always loved baking, ever since she was a little girl growing up in Brigham City, Utah. She loved the way the butter and sugar melted together in the heat of the oven, creating something deliciously aromatic and warm. She had started working at a local bakery when she was just eighteen years old, and had quickly become their star baker. Everyone who tasted her cakes and breads raved about how amazing they were. But even though Sherri had a talent for baking, she also had a lot of secrets buried deep within her. She had been raised by a strict Mormon family, who had instilled in her a sense of duty and obligation to her faith. But Sherri had always felt like there was something missing in her life, something that she couldn't quite put her finger on. One day, while she was working at the bakery, Sherri stumbled across a newspaper article that would change her life forever. The article was about a reform school for girls called Panorama School for Girls, which had recently been closed down due to allegations of abuse and mistreatment. As Sherri read the article, her heart raced faster and faster. She realized with a shock that she had attended that very school when she was a young teenager, and that she had been subjected to the same horrific abuse and mistreatment that the other girls had reported. Sherri felt traumatized by the memories flooding back to her. She had tried so hard to forget about her time at Panorama, to bury the memories and move on with her life. But now, faced with the reality of what had happened, she knew that she had to do something. Sherri quit her job at the bakery and started doing some research on Panorama. She read everything she could find about the school and the people who had run it, and she reached out to some of the other girls who had been there with her. Together, Sherri and the other girls formed a support group to help each other heal from the trauma they had experienced at Panorama. They went public with their stories, and eventually, the people responsible for running the school were brought to justice. Through her experience, Sherri found a new sense of purpose in her life. She started a nonprofit bakery, where she could use her baking skills to help other women who had experienced trauma and abuse. And she continued to use her voice to speak out against injustice, making sure that what happened at Panorama would never be forgotten."
    },
    {
        "prompt": "Paraphrase this in a comforting tone: Good afternoon today is april 9th the time here is 05:05 pm and im recording this Voice note again I woke up at 10 pm had brushed up wasn't really feeling well as my stomach was making me feel uncomfortable i had breakfast i did my prayers as today is Resurrections Sunday I hope prayers went well for you as well I hope you have eaten on time and you are keeping well i got a call from mom saying she reached native on 3pm the train was on time and there was no delays dad came to pick her up and help with luggages I still haven't received any Job updates so far as i'm still looking for more still trying feels different this time as I dont really get any notifications and calls from you these days now its going to be like this for 2 months I still wait for the messenger to make a tune to just to see if it's you I miss you a lot and I believe by tomorrow you have work and you won't be getting much time as you'll be very busy with schedules and work and you'll be under continuous work and less rest so it's alright make sure you are getting enough rest and taking good care of yourself and I'll be around for you waiting for your messages or reply and i know i have to send you the letter but now i don't really feel like sending it as i dont know how it might make you feel I dont want you to have like mixed thoughts and be like no you are hurting yourself and give up and move on and all that but i just cant think of doing any of those letter is to make you feel the love for you from me and how much i care i know you dont feel the same way towards me but i just want to keep it this way as i feel it might be the best way for me to cope. and make myself feel okay.",
        "response": "Hello there, it's nice to talk to you today. It's April 9th and the time is 5:05 pm. I hope you're doing well and have taken care of yourself. I had a bit of a rough morning with some stomach discomfort, but everything is alright now. I hope your prayers went well for Resurrection Sunday. I received a call from my mom, and she arrived safely at her destination. I am still waiting for job updates, but I am trying my best to stay positive. I miss you a lot, and it's tough not hearing from you much lately, but I understand that you're busy with work. Please make sure to take good care of yourself and know that I am always here for you whenever you need me. I was planning on sending you a letter, but now I'm not sure if it's the best thing to do. I just want you to know how much I care about you, even if our feelings aren't mutual. I'm doing my best to cope and hope that everything works out for the best.",
    },
    {
        "prompt": "write a 50 words A2 description of a fantasy animal half elephant half mouse",
        "response": "This imaginary creature is a unique blend of two worlds, half-elephant and half-mouse. It has the distinct features of both animals, with fuzzy ears and a long trunk. Despite its small size, it possesses the mighty strength of an elephant and the agility of a mouse."
    },
    {
        "prompt": "Write a very long, elaborate, descriptive and detailed shooting script, including a background and dialogues, for a 1990s Sitcom comic scene that centers around a food fight between female characters . You are free to choose the setting, scenario (it should make sense) and characters (give them names, and describe their appearance and clothing in detail).",
        "response": """Title: ``Table Manners'' Setting: The scene takes place at an upscale restaurant called \"Caviar \& Co.\". The interior design includes fancy chandeliers, elegant draperies, and vintage artwork displayed on the walls. Each table is neatly arranged with white tablecloths, fine china, and silverware, along with a centerpiece of a vibrant bouquet of flowers. Soft classical music plays in the background, creating a sophisticated ambience. Characters: 1) Emma: A woman in her early 30s. She wears a crisp white shirt with a black pencil skirt paired with classic black pumps. She has an air of sophistication and confidence. Her hair is elegantly styled with soft curls, and her makeup is modest and pristine. 2) Jane: Emma's sister, a woman in her late 20s. Jane dons a bright red dress with a sweetheart neckline and a playful hemline that ends just above her knees. Her hair is styled into a messy updo, and her makeup is bold with winged eyeliner and red lipstick. Jane appears to be the life of the party, always looking for fun and mischief. 3) Cecily: Emma and Jane's cousin, a woman in her mid-20s. She wears a flowy, pastel-colored dress with matching pumps. Cecily exudes an innocent sweetness with her rosy cheeks, natural makeup, and neatly braided hair. Her eyes gleam with a touch of naivety. Scene: The three women sit at a table in the restaurant, engaged in conversation while browsing through the menu. Waiters are busy around them attending to other tables. Emma: (Eyes her menu intently) I'm thinking of trying the poached salmon. How about you, Jane? Jane: (Mischievous grin) I'm feeling daring tonight. I'm going all in for the steak, rare! Cecily: (Giggles) You two are so adventurous. I think I'll just have the simple Caesar salad. Emma turns to signal the waiter to place their orders. Just as she does, Jane reaches into her purse and takes out a small squirt gun, discreetly aiming it at Cecily's plate. Jane: (Whispers) Hey, Cecily! Catch! Jane squirts a little ketchup from the gun onto Cecily's plate, creating a sudden commotion. Cecily: (Shocked) Jane! What in the world are you doing? Emma: (Disapproving look) Jane, this is not the place for your ridiculous pranks. Jane: (Unapologetic) Oh, come on, you two. Loosen up! It's just a little fun. Emboldened by Jane's mischief, Cecily unexpectedly grabs a spoonful of mashed potatoes from her plate and playfully flings it at Jane. Cecily: (Giggling) How about this for loosening up? Emma: (Surprised but amused) Cecily, I never thought you'd partake in something like this! As the mashed potatoes land on Jane's dress, she gasps dramatically and then smirks. Jane: (Feigning anger) Oh, you're on, young lady! Jane retaliates by tossing a piece of bread at Cecily, hitting her squarely on the forehead. Cecily laughs hysterically, pushing Emma to join in. Emma rolls her eyes but grabs a dollop of whipped cream from a nearby dessert and hurls it at Jane. Emma: If you can't beat 'em, join 'em. The three women begin an all-out food fight, grabbing anything on the table within reach - peas, filet mignon, even strings of spaghetti - and launching it at each other, laughing all the while. The chaos escalates, garnering the attention of other patrons and the restaurant staff. A waiter rushes over in an attempt to restore order. Waiter: (Trying to maintain composure) Ladies! Please! This is an exclusive establishment! I must insist you refrain from this... behavior immediately. The waiter's efforts are cut short as he is accidentally hit in the face with a spoonful of sauce from Emma's hand. Emma gasps, and Jane and Cecily try to stifle their laughter. Emma: (Nervously) Oh! I am so terribly sorry, sir. We got carried away. It won't happen again, I promise. Cecily: (Whispers to Jane) I can't believe we just did that. It was thrilling! Jane: (Whispers back) Isn't life more exciting when we let loose a little? The three women share a triumphant chuckle and nod in agreement as they begin to clean up the mess they created, with apologetic smiles directed towards the staff and other patrons. FADE OUT."""
    },
    {
        "prompt": "Rewrite the quote as the CEO of Akaike Technologies who is happy to have won Analytics India Certification for Great Firm for Data Scientists, 'Before starting Akaike I was researcher and a data scientist myself. That helped me create a culture of learning and experimentation and taking bold steps and all the other good stuff which any good data scientist looks for. I created a company where I would like to work as a data scientist myself.'",
        "response": """As the CEO of Akaike Technologies, I am thrilled to receive the Analytics India Certification for a Great Firm for Data Scientists. My background as a researcher and data scientist allowed me to establish a culture of continuous learning, experimentation, and bold initiatives that data scientists seek. Essentially, I built a company where I would personally love to work as a data scientist."""
    },
    {
        "prompt": "Write a message for Happy Anniversary",
        "response": """Happy Anniversary to the most loving couple I know! You both never cease to amaze me with your love and commitment. I wish you a day filled with joy and love as you celebrate this milestone in your relationship. May your love continue to grow stronger with each passing year. Congratulations and cheers to many more years of love and togetherness!"""
    }
]
    


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="news_bbc.jsonl",
        help="The dataset file",
    )
    parser.add_argument(
        "--llm_config_file",
        type=str,
        default="./configs/llm.yaml",
        help="The config file of evaluating large language models (LLMs).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["news", "paper", "patent", "poem"],
        help="The dataset name. (e.g. news)",
    )
    parser.add_argument(
        "--output_result_dir",
        type=str,
        default="./result/",
        help="The output directory of the figure.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="output",
        help="The output name of the figure.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Whether to resume the inference process.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="The number of samples to evaluate.",
    )

    args = parser.parse_args()
    return args


def cal_individual_score(
    id, outputs1, outputs2, target_ids, prompt_length, num_words=None
):
    logits1 = outputs1.logits.detach().cpu()
    logits2 = outputs2.logits.detach().cpu()

    nll1 = outputs1.loss.item()
    nll2 = outputs2.loss.item()

    probabilities = F.softmax(logits1[:, :-1], dim=-1)
    probabilities2 = F.softmax(logits2[:, prompt_length:-1], dim=-1)

    collaborative_one_hot = F.one_hot(
        torch.tensor(target_ids), num_classes=probabilities.size(-1)
    ).float()

    entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
    entropy2 = -torch.sum(probabilities2 * torch.log(probabilities2), dim=-1)

    ce = -torch.sum(collaborative_one_hot * torch.log(probabilities), dim=-1)
    ce2 = -torch.sum(collaborative_one_hot * torch.log(probabilities2), dim=-1)

    human_percent = (
        (torch.sum(entropy) - torch.sum(entropy2)) / torch.sum(entropy)
    ).item()
    human_percent2 = ((torch.sum(ce) - torch.sum(ce2)) / torch.sum(ce)).item()
    human_percent3 = (nll1 - nll2) / nll1

    rslt = {
        "id": id,
        "en": entropy.squeeze().cpu().numpy().tolist(),
        "en2": entropy2.squeeze().cpu().numpy().tolist(),
        "ce": ce.squeeze().cpu().numpy().tolist(),
        "ce2": ce2.squeeze().cpu().numpy().tolist(),
        "nll1": nll1,
        "nll2": nll2,
        "human_percent": human_percent,
        "human_percent2": human_percent2,
        "human_percent3": human_percent3,
        "num_words": num_words,
    }
    return rslt


def construct_prompt(example, model_id):
    conv_template = fastchat.model.get_conversation_template(model_id)

    system_prompt = "You are a helpful assistant."
    user_prompt = example["prompt"]

    conv_template.set_system_message(system_prompt)
    conv_template.append_message(conv_template.roles[0], user_prompt)
    conv_template.append_message(conv_template.roles[1], None)

    example["message"] = conv_template.get_prompt()
    return example


def get_prompt_dict(cases, model_id):
    data_dict = defaultdict(list)
    for idx, case in enumerate(cases):
        data_dict["id"].append(idx)
        for key, value in case.items():
            data_dict[key].append(value)

    dataset = Dataset.from_dict(data_dict)

    processed_dataset = dataset.map(
        partial(
            construct_prompt,
            model_id=model_id,
        ),
        desc="Processing datasets.",
    )

    prompt_dict = {
        index: prompt
        for prompt, index in zip(
            processed_dataset["message"],
            processed_dataset["id"],
        )
    }
    return prompt_dict


def cal_loss_wo_rewrite(
    cases,
    model_id,
    output_name,
    output_result_dir="./result/",
    device="cuda",
    resume=False,
):
    prompt_dict = get_prompt_dict(cases, model_id)

    file_path = output_result_dir + output_name + "_ces.json"
    if resume and os.path.exists(file_path):
        with open(file_path, "r") as file:
            ces = json.load(file)
    else:
        out = []
        model, tokenizer = load_model(model_id, device)
        
        for idx, sample in tqdm(enumerate(cases)):
            response = sample["response"]
            
            with torch.no_grad():
                extracted_text = response + tokenizer.eos_token

                input_ids = torch.tensor(
                    np.array([tokenizer(extracted_text).input_ids])
                ).to(device)

                target_ids = input_ids.clone()
                sep = 1
                target_ids[:, :sep] = -100
                outputs1 = model(input_ids, labels=target_ids)

            with torch.no_grad():
                prompt = prompt_dict[idx]

                prompt_tokens = tokenizer(
                    prompt, add_special_tokens=False
                ).input_ids
                response_tokens = tokenizer(
                    response, add_special_tokens=False
                ).input_ids

                extracted_text = (
                    [tokenizer.bos_token_id]
                    + prompt_tokens
                    + response_tokens
                    + [tokenizer.eos_token_id]
                )

                input_ids = torch.tensor([extracted_text]).to(device)
                target_ids = input_ids.clone()
                sep = len(prompt_tokens) + 1
                target_ids[:, :sep] = -100
                outputs2 = model(input_ids, labels=target_ids)

            rslt = cal_individual_score(
                idx,
                outputs1,
                outputs2,
                response_tokens + [tokenizer.eos_token_id],
                len(prompt_tokens),
                num_words=sample.get("num_words", None),
            )
            out.append(rslt)

        with jsonlines.open(
            output_result_dir + output_name + "_ces_id.jsonl", "w"
        ) as writer:
            writer.write_all(out)
    


def load_model(model_id, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )

    model.eval()
    print(model.__class__)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    return model, tokenizer


if __name__ == "__main__":
    args = parse_args()

    with open(args.llm_config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    Path(args.output_result_dir).mkdir(parents=True, exist_ok=True)

    cal_loss_wo_rewrite(
        cases,
        config["model_name"],
        args.output_name + "_wo_rewrite",
        output_result_dir=args.output_result_dir,
        resume=args.resume,
    )
