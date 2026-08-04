import os
import pickle
import json
import subprocess

videos = ['02I8Ad7qkjQ', '0bJV_IK_MPA', '0EH9HggXhUo', '0FmwpU0rk80']
out_dir = 'data/union_test'
os.makedirs(out_dir, exist_ok=True)

for vid in videos:
    pkl_path = f'data/memory_graphs/web/{vid}_audio_dev.pkl'
    if not os.path.exists(pkl_path):
        continue

    print(f"\nProcessing {vid}...")
    with open(pkl_path, 'rb') as f:
        graph = pickle.load(f)

    mappings = graph.character_mappings

    # only care about characters that have more than 1 voice tag to test union effectiveness
    union_chars = {}
    for char, tags in mappings.items():
        voice_tags = [t for t in tags if t.startswith('voice_')]
        if len(voice_tags) > 1:
            union_chars[char] = voice_tags

    if not union_chars:
        print(f"No multiple voice characters in {vid}")
        continue

    v_dir = os.path.join(out_dir, vid)
    os.makedirs(v_dir, exist_ok=True)

    for char, voice_tags in union_chars.items():
        c_dir = os.path.join(v_dir, char)
        os.makedirs(c_dir, exist_ok=True)
        print(f"  {char}: {voice_tags}")

        for v_tag in voice_tags:
            node_id = int(v_tag.split('_')[1])
            node = graph.nodes[node_id]
            content = node.metadata['contents'][0] if node.metadata.get('contents') else ""

            # Find the clip_id by finding connected episodic/semantic nodes
            clip_id = None
            for u, v in graph.edges:
                if u == node_id and graph.nodes[v].type in ['episodic', 'semantic']:
                    clip_id = graph.nodes[v].metadata.get('timestamp')
                    if clip_id is not None:
                        break
                elif v == node_id and graph.nodes[u].type in ['episodic', 'semantic']:
                    clip_id = graph.nodes[u].metadata.get('timestamp')
                    if clip_id is not None:
                        break

            if clip_id is None:
                print(f"  Cannot find clip_id for {v_tag}")
                continue

            with open(os.path.join(c_dir, f"{v_tag}.txt"), "w") as f:
                f.write(f"Clip: {clip_id}\nContent: {content}\n")

            src_mp4 = f"data/clips/web/{vid}/{clip_id}.mp4"
            dst_mp4 = os.path.join(c_dir, f"{v_tag}_clip{clip_id}.mp4")

            if os.path.exists(src_mp4):
                subprocess.run(f"cp {src_mp4} {dst_mp4}", shell=True)
            else:
                print(f"  Source clip missing: {src_mp4}")

print("Extraction complete.")
