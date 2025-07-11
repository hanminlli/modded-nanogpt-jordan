import os
import sys
from huggingface_hub import hf_hub_download

# Download the GPT-2 tokens of Fineweb10B from huggingface. This
# saves about an hour of startup time compared to regenerating them.
# This is already-tokenized Fineweb10B, which is in ready-to-train binary format
def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), 'fineweb10B')
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
                        repo_type="dataset", local_dir=local_dir)
        
if __name__ == "__main__":
    get("fineweb_val_%06d.bin" % 0) # name: fineweb_val_000000.bin
    num_chunks = 103 # full fineweb10B. Each chunk is 100M tokens

    if len(sys.argv) >= 2: # we can pass an argument to download less
        num_chunks = int(sys.argv[1])

    for i in range(1, num_chunks+1):
        get("fineweb_train_%06d.bin" % i) # From 000001 to 000103
