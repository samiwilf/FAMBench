import pathlib

LOG_PATH = "/home/ubuntu/repos/FAMBench/benchmarks/dlrm/ootb/"
SETTING = 3
print('*'.center(40, '*'))
print(f"  RUNNING SETTING {SETTING}  ".center(40, '*'))
print('*'.center(40, '*'))

SAVE_DEBUG_DATA = True if SETTING != 5 else False

DENSE_LOG_FILE = pathlib.Path(LOG_PATH + "s" + str(SETTING) + "_DENSE.txt")
SPARSE_LOG_FILE = pathlib.Path(LOG_PATH + "s" + str(SETTING) + "_SPARSE.txt")
D_OUT_LOG_FILE = pathlib.Path(LOG_PATH + "s" + str(SETTING) + "_D_OUT.txt")
E_OUT_LOG_FILE = pathlib.Path(LOG_PATH + "s" + str(SETTING) + "_E_OUT.txt")
C_OUT_LOG_FILE = pathlib.Path(LOG_PATH + "s" + str(SETTING) + "_C_OUT.txt")

if SETTING == 1:
    LOG_FILE = "Losses_setting1_simplestNN_oss.txt"   
    INT_FEATURE_COUNT = 1
    DAYS = 1
    ARGV = ["--data-generation=random",
        "--arch-embedding-size=1",
        "--arch-sparse-feature-size=4", 
        "--arch-mlp-bot=1-4", 
        "--arch-mlp-top=1-1",         
        "--mini-batch-size=1", 
        "--num-batches=10", 
        "--use-gpu", 
        "--memory-map", 
        "--loss-function=bce", 
        "--test-mini-batch-size=1", 
        "--print-freq=1024", 
        "--print-time", 
        "--nepoch=1", 
        "--max-ind-range=40000000", 
        "--learning-rate=1.0", 
        "--round-targets=True", 
        "--test-num-workers=16", 
        "--test-freq=30000", 
        "--data-set=terabyte", 
        "--processed-data-file=/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/", 
        "--raw-data-file=/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/day", 
    ]

if SETTING == 2:
    LOG_FILE = "Losses_setting2_simplestNN_oss.txt"   
    INT_FEATURE_COUNT = 1
    DAYS = 1
    ARGV = ["--data-generation=random",
        "--arch-sparse-feature-size=4", 
        "--arch-embedding-size=1",
        "--arch-mlp-top=1-1", 
        "--arch-mlp-bot=1-4", 
        "--mini-batch-size=8388608",
        "--num-batches=10", 
        "--use-gpu", 
        "--use-fbgemm-gpu",
        "--memory-map", 
        "--loss-function=bce", 
        "--test-mini-batch-size=1", 
        "--print-freq=1024", 
        "--print-time", 
        "--nepoch=1", 
        "--max-ind-range=40000000", 
        "--learning-rate=1.0", 
        "--round-targets=True", 
        "--test-num-workers=16", 
        "--test-freq=30000", 
        "--data-set=terabyte", 
        "--processed-data-file=/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/", 
        "--raw-data-file=/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/day", 
        ] 

if SETTING == 3:
    LOG_FILE = "s3_oss.txt"   
    INT_FEATURE_COUNT = 1
    DAYS = 1
    ARGV = [
        "--mini-batch-size=2048", 
        "--arch-sparse-feature-size=128", 
        "--arch-embedding-size=45833188-36746-17245",
        "--arch-mlp-bot=1-512-256-128", 
        "--arch-mlp-top=1024-1024-512-256-1",
        "--data-generation=random",
        "--learning-rate=1.0", 
        "--num-batches=10", 
        "--use-gpu", 
        #"--use-torch2trt-for-mlp",
        #"--inference-only",
        #"--use-fbgemm-gpu", 
        "--loss-function=bce", 
        "--nepoch=1", 
        "--max-ind-range=40000000", 
        "--round-targets=True", 
        "--test-num-workers=16", 
        "--raw-data-file=/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/day", 
        "--processed-data-file=/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/", 
        "--memory-map", 
        "--data-set=terabyte", 
        "--print-freq=1024", 
        "--print-time", 
        "--test-mini-batch-size=1", 
        "--test-freq=30000", 
    ]

if SETTING == 4:
    LOG_FILE = "s4_losses_day_0.txt"
    INT_FEATURE_COUNT = 13 #1 #13
    CAT_FEATURE_COUNT = 26 #2
    DAYS = 1#24   
    ARGV = ["--data-generation=dataset",
        "--data-set=terabyte", 
        "--mini-batch-size=2048", 
        #"--arch-embedding-size= this is read from file", 
        "--arch-mlp-bot=13-512-256-128", 
        "--arch-mlp-top=1024-1024-512-256-1", 
        "--arch-sparse-feature-size=128", 
        "--learning-rate=1.0", 
        "--num-batches=10", 
        "--use-gpu", 
        "--raw-data-file=/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/day", 
        "--processed-data-file=/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/", 
        "--memory-map", 
        "--loss-function=bce", 
        "--test-mini-batch-size=1", 
        "--print-freq=1024", 
        "--print-time", 
        "--nepoch=1", 
        "--max-ind-range=40000000", 
        "--round-targets=True", 
        "--test-num-workers=16", 
        "--test-freq=30000", 
    ]

for a in ARGV:
    if "--arch-embedding-size" in a:
        CAT_FEATURE_COUNT = len(a[a.find('=')+1:].split('-'))
        break

[print(a) for a in ARGV]

LOG_FILE = pathlib.Path(LOG_PATH + LOG_FILE)
for f in [LOG_FILE, DENSE_LOG_FILE, SPARSE_LOG_FILE, D_OUT_LOG_FILE, E_OUT_LOG_FILE, C_OUT_LOG_FILE]:
    if f.is_file():
        f.unlink()

    # simplest NN test
    #argv = ['--pin_memory', '--batch_size', '1', '--epochs', '1', '--num_embeddings_per_feature', '1', '--embedding_dim', '4', '--dense_arch_layer_sizes', '4', '--over_arch_layer_sizes', '1,1', '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M', '--learning_rate', '1.0']

    # simplest 2 GPU NN test
    #argv = ['--pin_memory', '--batch_size', '2', '--epochs', '1', '--num_embeddings_per_feature', '1,1', '--embedding_dim', '4', '--dense_arch_layer_sizes', '4', '--over_arch_layer_sizes', '1,1', '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M', '--learning_rate', '1.0']

    # real embedding sizes, but fake data and small mlp layers.
    #argv = ['--pin_memory', '--batch_size', '2048', '--epochs', '1', '--num_embeddings_per_feature', '45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35', '--embedding_dim', '128', '--dense_arch_layer_sizes', '128', '--over_arch_layer_sizes', '1,1', '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M', '--learning_rate', '0.01']

    # real run except limiting test, val, and train batches
    #argv = ['--seed','1','--limit_test_batches', '1', '--limit_val_batches', '1', '--limit_train_batches', '5', '--pin_memory', '--batch_size', '2048', '--epochs', '1', '--num_embeddings_per_feature', '45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35', '--embedding_dim', '128', '--dense_arch_layer_sizes', '512,256,128', '--over_arch_layer_sizes', '1024,1024,512,256,1', '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M', '--learning_rate', '1.0']
    
    # real run
    #argv = ['--pin_memory', '--batch_size', '2048', '--epochs', '1', '--num_embeddings_per_feature', '45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35', '--embedding_dim', '128', '--dense_arch_layer_sizes', '512,256,128', '--over_arch_layer_sizes', '1024,1024,512,256,1', '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M', '--learning_rate', '1.0']
    
    # modify embedding tables to be 1 vector per table.
    #argv = ['--pin_memory', '--batch_size', '2048', '--epochs', '1', '--num_embeddings_per_feature', '1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1', '--embedding_dim', '128', '--dense_arch_layer_sizes', '512,256,128', '--over_arch_layer_sizes', '1024,1024,512,256,1', '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M', '--learning_rate', '1.0']

    # real run, but using Shabab's 1tb_numpy data
    #argv = ['--pin_memory', '--batch_size', '2048', '--epochs', '1', '--num_embeddings_per_feature', '45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35', '--embedding_dim', '128', '--dense_arch_layer_sizes', '512,256,128', '--over_arch_layer_sizes', '1024,1024,512,256,1', '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo/1tb_numpy/', '--learning_rate', '1.0']
