"""Convert official ViTPose COCO-WholeBody (133 kp) weights (easy_ViTPose mirror) into a HF VitPoseForPoseEstimation.
-> models/vitpose_weights/vitpose-<size>-wholebody-hf/   usage: python scripts/pose/convert_wholebody.py [l|h|b]"""
import os, sys
import torch
from huggingface_hub import hf_hub_download
from transformers import (VitPoseConfig, VitPoseBackboneConfig, VitPoseForPoseEstimation, VitPoseImageProcessor)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
size = sys.argv[1] if len(sys.argv) > 1 else 'l'
DIMS = {'b': (768, 12, 12), 'l': (1024, 24, 16), 'h': (1280, 32, 16)}
hid, layers, heads = DIMS[size]

src = hf_hub_download('JunkyByte/easy_ViTPose', f'torch/wholebody/vitpose-{size}-wholebody.pth',
                      local_dir=os.path.join(ROOT, 'models', 'vitpose_weights'))
sd = torch.load(src, map_location='cpu', weights_only=False)['state_dict']

bc = VitPoseBackboneConfig(hidden_size=hid, num_hidden_layers=layers, num_attention_heads=heads,
                           intermediate_size=4 * hid, image_size=[256, 192], patch_size=[16, 16],
                           qkv_bias=True, out_indices=[layers])
cfg = VitPoseConfig(backbone_config=bc, use_simple_decoder=False, num_labels=133)
model = VitPoseForPoseEstimation(cfg).eval()

new = {'backbone.embeddings.position_embeddings': sd['backbone.pos_embed'],
       'backbone.embeddings.patch_embeddings.projection.weight': sd['backbone.patch_embed.proj.weight'],
       'backbone.embeddings.patch_embeddings.projection.bias': sd['backbone.patch_embed.proj.bias'],
       'backbone.layernorm.weight': sd['backbone.last_norm.weight'],
       'backbone.layernorm.bias': sd['backbone.last_norm.bias']}
for i in range(layers):
    s, d = f'backbone.blocks.{i}.', f'backbone.encoder.layer.{i}.'
    for w in ('weight', 'bias'):
        q, k, v = sd[s + f'attn.qkv.{w}'].chunk(3, 0)
        new[d + f'attention.attention.query.{w}'] = q
        new[d + f'attention.attention.key.{w}'] = k
        new[d + f'attention.attention.value.{w}'] = v
        new[d + f'attention.output.dense.{w}'] = sd[s + f'attn.proj.{w}']
        new[d + f'mlp.fc1.{w}'] = sd[s + f'mlp.fc1.{w}']
        new[d + f'mlp.fc2.{w}'] = sd[s + f'mlp.fc2.{w}']
        new[d + f'layernorm_before.{w}'] = sd[s + f'norm1.{w}']
        new[d + f'layernorm_after.{w}'] = sd[s + f'norm2.{w}']
h = 'keypoint_head.'
new['head.deconv1.weight'] = sd[h + 'deconv_layers.0.weight']
new['head.deconv2.weight'] = sd[h + 'deconv_layers.3.weight']
for a, b in (('batchnorm1', 'deconv_layers.1'), ('batchnorm2', 'deconv_layers.4')):
    for w in ('weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked'):
        new[f'head.{a}.{w}'] = sd[f'{h}{b}.{w}']
new['head.conv.weight'] = sd[h + 'final_layer.weight']
new['head.conv.bias'] = sd[h + 'final_layer.bias']

missing, unexpected = model.load_state_dict(new, strict=False)
print('missing:', missing, 'unexpected:', unexpected)
assert not unexpected and all('layernorm' not in m for m in missing), missing
out = os.path.join(ROOT, 'models', 'vitpose_weights', f'vitpose-{size}-wholebody-hf')
model.save_pretrained(out)
VitPoseImageProcessor.from_pretrained('usyd-community/vitpose-plus-large').save_pretrained(out)
print('saved', out)
