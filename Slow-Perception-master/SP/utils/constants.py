CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "log"

IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"

DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_BOX_TOKEN = "<box>"

DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'



CONVERSATION_DATA = {

    'data1': {
        'images': 'SP-1/train_sp1/sub1/imgs/',
        'annotations': 'SP-1/jihe_slow_4ruler1.json',
    },

    'data2': {
        'images': 'SP-1/train_sp1/sub2/imgs/',
        'annotations': 'SP-1/jihe_slow_4ruler2.json',
    },

    # 'data1': {
    #     'images': 'SP-1/train_sp1/sub1/imgs/',
    #     'annotations': 'SP-1/jihe_slow_baseline1.json',
    # },
    #
    # 'data2': {
    #     'images': 'SP-1/train_sp1/sub2/imgs/',
    #     'annotations': 'SP-1/jihe_slow_baseline1.json',
    # },



}