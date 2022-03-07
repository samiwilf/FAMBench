

LOG_PATH = "/home/ubuntu/repos/FAMBench/benchmarks/dlrm/ootb/"
SETTING = 3

if SETTING == 1:
    LOG_FILE = "Losses_setting1_simplestNN_oss.txt"   
    INT_FEATURE_COUNT = 1
    CAT_FEATURE_COUNT = 1
    DAYS = 1
    ARGV = ["--data-generation=random",
        "--data-set=terabyte", 
        "--raw-data-file=/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/day", 
        "--processed-data-file=/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/", 
        "--memory-map", 
        "--loss-function=bce", 
        "--mini-batch-size=1", 
        "--test-mini-batch-size=1", 
        "--print-freq=1024", 
        "--print-time", 
        "--use-gpu", 
        "--nepoch=1", 
        "--arch-sparse-feature-size=4", 
        "--arch-mlp-bot=1-4", 
        "--arch-mlp-top=1-1", 
        "--max-ind-range=40000000", 
        "--learning-rate=1.0", 
        "--num-batches=10", 
        "--round-targets=True", 
        "--test-num-workers=16", 
        "--test-freq=30000", 
        "--arch-embedding-size=1"]

if SETTING == 2:
    LOG_FILE = "Losses_setting2_simplestNN_oss.txt"   
    INT_FEATURE_COUNT = 1
    CAT_FEATURE_COUNT = 1
    DAYS = 1
    ARGV = ["--data-generation=random",
        "--data-set=terabyte", 
        "--raw-data-file=/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/day", 
        "--processed-data-file=/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/", 
        "--memory-map", 
        "--loss-function=bce", 
        "--test-mini-batch-size=1", 
        "--print-freq=1024", 
        "--print-time", 
        "--use-gpu", 
        "--nepoch=1", 
        "--arch-sparse-feature-size=4", 
        "--arch-mlp-bot=1-4", 
        "--arch-mlp-top=1-1", 
        "--max-ind-range=40000000", 
        "--learning-rate=1.0", 
        "--num-batches=10", 
        "--round-targets=True", 
        "--test-num-workers=16", 
        "--test-freq=30000", 
        "--arch-embedding-size=1",
        "--mini-batch-size=8388608",
        "--use-fbgemm-gpu",
        ] 

if SETTING == 3:
    LOG_FILE = "Losses_setting3_simplestNN_oss.txt"   
    INT_FEATURE_COUNT = 1
    CAT_FEATURE_COUNT = 1
    DAYS = 1
    ARGV = ["--data-generation=random",
        "--data-set=terabyte", 
        "--mini-batch-size=128", 
        "--arch-embedding-size=16-16-16-16",
        "--arch-mlp-bot=1-4", 
        "--arch-mlp-top=1-1", 
        "--arch-sparse-feature-size=4", 
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

if SETTING == 33:
    LOG_FILE = "/home/ubuntu/repos/torchrec/examples/dlrm/Losses_day_0_single_sample_.txt"
    INT_FEATURE_COUNT = 13 #1 #13
    CAT_FEATURE_COUNT = 26 #2
    DAYS = 1#24   
    ARGV = ['--pin_memory', 
        '--batch_size', '2048', 
        '--epochs', '1', 
        '--num_embeddings_per_feature', '45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35', 
        '--embedding_dim', '128', 
        '--dense_arch_layer_sizes', '512,256,128', 
        '--over_arch_layer_sizes', '1024,1024,512,256,1', 
        '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M', 
        '--learning_rate', '1.0']         

LOG_FILE = LOG_PATH + LOG_FILE

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
