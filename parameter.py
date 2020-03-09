RECORD_NUM = 1

TOTAL_FRAME = 90
FIRST_FRAME = 10
SECOND_FRAME = 13

ACTION_SIZE = 4
FIRST_TRAIN = True

AVERAGE_TIME = 1 # use average value to decrease of the effect of randomness

LR_A = 0.00001    # learning rate for actor
LR_C = 0.0002     # learning rate for critic

STATE_SIZE_X = 64
STATE_SIZE_Y = 64

GAMMA = 0.9     # reward discount in TD error
AUGMENT_RATE = 0.1


############################################################################################
############################################################################################
############################################################################################

REPOSITORY_PATH = "/home/smj/OnAVOS_new" # the path refers the root path of repository


############# DAVIS 2016 ###################

# Local
DATSET_PATH = "/home/smj/DataSet/DAVIS/DAVIS/JPEGImages/480p/"
# Remote
#DATSET_PATH = "/Data_HDD/smj_data/DAVIS/JPEGImages/480p/"

#NAME_PAIR_PATH = "./name_pair/davis2016/davis2016-full.txt"
#NAME_SEQ = "./name_seq/davis2016/davis2016-full.txt"

#NAME_PAIR_PATH = "./name_pair/davis2016/davis2016-mini.txt"
#NAME_SEQ = "./name_seq/davis2016/davis2016-mini.txt"

#NAME_PAIR_PATH = "./name_pair/davis2016/davis2016-fixed2.txt"
#NAME_SEQ = "./name_seq/davis2016/davis2016-fixed.txt"

#NAME_PAIR_PATH = "./name_pair/davis2016/davis2016-train.txt"
#NAME_SEQ = "./name_seq/davis2016/davis2016-train.txt"


############### YoutubeVOS ########################3

# Local
#DATSET_PATH = "/home/smj/DataSet/YoutubeVOS/JPEGImages/"
#LABEL_PATH = "/home/smj/DataSet/YoutubeVOS/Annotations/"

# Remote
#DATSET_PATH = "/Data_HDD/smj_data/YoutubeVOS/JPEGImages/"


#NAME_PAIR_PATH = "./name_pair/YoutubeVOS/train_parent_test"
#NAME_SEQ = "./name_seq/YoutubeVOS/name_test.txt"

#NAME_PAIR_PATH = "./name_pair/YoutubeVOS/train_parent_fixed"
#NAME_SEQ = "./name_seq/YoutubeVOS/name_fixed.txt"

#NAME_PAIR_PATH = "./name_pair/YoutubeVOS/train_parent_1"
#NAME_SEQ = "./name_seq/YoutubeVOS/name_1.txt"

# MODEL

NAME_PAIR_PATH = "./pair"
NAME_SEQ = "./seq"


