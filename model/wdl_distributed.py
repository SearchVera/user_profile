import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import Hashing, CategoryEncoding, Dense, Dropout, Layer, Embedding
from tensorflow.keras.models import load_model
from tensorflow.keras.experimental import SidecarEvaluator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras import regularizers
from absl import flags, app
import datetime
import json
import os
import re

##参数定义
FLAGS = flags.FLAGS
flags.DEFINE_string("mode", "train", "train/predict")
flags.DEFINE_string("version", "v1", "version")
flags.DEFINE_string("worker_hosts", "", "worker_hosts")
flags.DEFINE_string("ps_hosts", "", "ps_hosts")
flags.DEFINE_string("job_name", "", "job_name")
flags.DEFINE_integer("task_index", -1, "task_index")
flags.DEFINE_bool("use_evaluator", True, "use_evaluator")
flags.DEFINE_string("ckpt_version", "1", "ckpt_version")
flags.DEFINE_string("day", "1", "day")

# data config
column_names = ['id', 'label', 'wide_fea1', 'wide_fea2', 'deep_fea1', 'deep_fea2']
label_name = 'label'
output_types = ({'id': tf.string,
                 'wide_fea1': tf.string,
                 'wide_fea2': tf.string,
                 'deep_fea1': tf.string,
                 'deep_fea2': tf.string
                 }, tf.float32)

base_dir = 'hdfs://*/'
train_data_dir = base_dir + 'trainset/*'
eval_data_dir = base_dir + 'evalset/*'
pred_data_dir = base_dir + 'predset/*'

# model config
model_dir = 'hdfs://*'
result_dir = 'hdfs://*'
batch_size = 512
num_epochs = 20
steps_per_epoch = 400
shuffle = True
shuffle_buffer_size = 10000
shuffle_seed = 23
hidden_units = [128, 64]
learning_rate = 0.01
#
hash_layer_conf = {
    'wide_fea1': 10000,
    'wide_fea2': 10000
}

#
# distribute config
NUM_WORKERS = 2
NUM_PS = 1


class MLP(Layer):
    def __init__(self, hidden_units, activation='relu', dropout=0.2, regularize=0.001):
        super(MLP, self).__init__()
        self.dnn_network = []
        for unit in hidden_units:
            self.dnn_network.append(
                Dense(units=unit, activation=activation, kernel_regularizer=regularizers.l2(regularize)))
            self.dnn_network.append(Dropout(dropout))

        self.dnn_network.append(Dense(1, activation=None))

    def call(self, inputs):
        x = inputs
        for net in self.dnn_network:
            x = net(x)
        return x


def gen_emb_layer(input, hash_layer, emb_layer, need_reduce=True):
    input_split = tf.strings.split(input, sep=',')
    input_hash = hash_layer(input_split)
    input_emb = emb_layer(input_hash)
    return tf.math.reduce_mean(input_emb, axis=1) if need_reduce else input_emb


def gen_onehot_layer(input, layers):
    input_split = tf.strings.split(input, sep=',')
    # hash
    input_hash = layers[0](input_split)
    # cate
    return layers[1](input_hash)


class WDL(tf.keras.Model):

    def __init__(self, hidden_units, hash_layer_conf, activation='relu', dropout=0.2, regularize=0.001):
        super().__init__()

        self.hash_layer_dict = {}
        for name, bins in hash_layer_conf.items():
            hash_layer = Hashing(num_bins=bins)
            cate_layer = CategoryEncoding(num_tokens=bins, output_mode="multi_hot")
            self.hash_layer_dict[name] = (hash_layer, cate_layer)

        # emb
        self.fea_hash = Hashing(num_bins=10000)
        self.fea_emb = Embedding(10000, 64)

        # deep layer
        self.mlp = MLP(hidden_units, activation, dropout, regularize)

        # wide layer
        self.linear_wide = Dense(1, activation=None)

    def call(self, inputs):
        # wide feature
        wide_fea1 = gen_onehot_layer(inputs['wide_fea1'], self.hash_layer_dict['wide_fea1'])
        wide_fea2 = gen_onehot_layer(inputs['wide_fea2'], self.hash_layer_dict['wide_fea2'])

        # deep feature
        deep_fea1 = gen_emb_layer(inputs['deep_fea1'], self.fea_hash, self.fea_emb)
        deep_fea2 = gen_emb_layer(inputs['deep_fea2'], self.fea_hash, self.fea_emb)

        # wide
        wide_input = tf.concat([wide_fea1, wide_fea2], axis=-1)
        wide_out = self.linear_wide(wide_input)

        # deep
        deep_input = tf.concat([deep_fea1, deep_fea2], axis=-1)
        deep_out = self.mlp(deep_input)

        # out
        outputs = tf.nn.sigmoid(0.5 * wide_out + 0.5 * deep_out)
        # outputs = tf.nn.sigmoid(wide_out)
        return outputs


def train(model_path):
    print('[start training]')
    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    if cluster_resolver.task_type in ("worker", "ps"):
        print('[this is worker/ps]')
        server = tf.distribute.Server(
            cluster_resolver.cluster_spec(),
            job_name=cluster_resolver.task_type,
            task_index=cluster_resolver.task_id,
            protocol="grpc")
        # Blocking the process that starts a server from exiting.
        server.join()
    elif cluster_resolver.task_type == 'evaluator':
        print('[this is evaluator]')
        model = WDL(hidden_units, hash_layer_conf)
        model.compile(metrics=[AUC()])
        dataset = tf.data.Dataset.from_generator(generator, output_types=output_types,
                                                 args=[eval_data_dir, False]).batch(batch_size)

        SidecarEvaluator(
            model=model,
            data=dataset,
            # dir for training-saved checkpoint
            checkpoint_dir=os.path.join(model_path, 'ckpt'),
            steps=200,  # Eval until dataset is exhausted
            max_evaluations=None,  # The evaluation needs to be stopped manually
            callbacks=[tf.keras.callbacks.TensorBoard(log_dir=os.path.join(model_path, 'log'))]
        ).start()
    else:
        print('[this is chief]')
        variable_partitioner = (
            tf.distribute.experimental.partitioners.MinSizePartitioner(
                min_shard_bytes=(256 << 10),
                max_shards=NUM_PS))
        strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver,
                                                                      variable_partitioner=variable_partitioner)
        with strategy.scope():
            model = WDL(hidden_units, hash_layer_conf)
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate,
                decay_steps=steps_per_epoch,
                decay_rate=0.5,
                staircase=True)
            model.compile(loss=BinaryCrossentropy(), optimizer=Adam(learning_rate=lr_schedule), metrics=[AUC()])

        def dataset_fn(input_context):
            dataset = tf.data.Dataset.from_generator(generator, output_types=output_types, args=[train_data_dir, False])
            dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
            dataset = dataset.repeat().shuffle(shuffle_buffer_size)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            return dataset

        input_options = tf.distribute.InputOptions(experimental_fetch_to_device=True)
        distributed_dataset = tf.keras.utils.experimental.DatasetCreator(dataset_fn, input_options=input_options)

        callbacks = [
            TensorBoard(log_dir=os.path.join(model_path, 'log')),
            ModelCheckpoint(filepath=os.path.join(model_path, 'ckpt', 'ckpt-{epoch}'), verbose=1,
                            save_weights_only=True)
        ]
        model.fit(distributed_dataset,
                  epochs=num_epochs,
                  steps_per_epoch=steps_per_epoch,
                  verbose=2,
                  callbacks=callbacks
                  )

        print('[training end]')


def predict(model_path, ckpt_version, result_path):
    print('[start predict]')
    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    if cluster_resolver.task_type != 'ps':
        model_path = os.path.join(model_path, 'ckpt', 'ckpt-{}'.format(ckpt_version))
        result_path = os.path.join(result_path, 'part-{:0>3d}'.format(FLAGS.task_index))
        print('[model path={}]\n[result path={}]'.format(model_path, result_path))

        model = WDL(hidden_units, hash_layer_conf)
        model.load_weights(model_path)
        model.compile(metrics=[AUC()])

        dataset = tf.data.Dataset.from_generator(generator, output_types=output_types,
                                                 args=[pred_data_dir, True]).batch(batch_size)

        idx = 0
        with tf.io.gfile.GFile(result_path, 'w') as f:
            for (data, label) in dataset:
                probs = model(data)
                ids = data['id']
                for id, prob in zip(ids.numpy().astype('str'), probs.numpy()):
                    print(id, prob[0], sep='\t', file=f)

                idx += 1
                if idx % 100 == 0:
                    print('worker={}, batch={}, num={}'.format(FLAGS.task_index, idx, idx * batch_size))

    print('[predict end]')


def gen_tf_config():
    worker_hosts = FLAGS.worker_hosts.split(',')
    ps_hosts = FLAGS.ps_hosts.split(',')
    use_evaluator = FLAGS.use_evaluator
    task_type = FLAGS.job_name
    task_index = FLAGS.task_index

    global NUM_WORKERS
    NUM_WORKERS = len(worker_hosts)
    global NUM_PS
    NUM_PS = len(ps_hosts)
    print('[num workers={},num ps={},task_type={},task_index={}]'.format(NUM_WORKERS, NUM_PS, task_type, task_index))

    tf_config = {'cluster': {}, 'task': {}}
    tf_config['cluster']['ps'] = ps_hosts
    tf_config['cluster']['chief'] = [worker_hosts[0]]

    if use_evaluator:
        tf_config['cluster']['worker'] = worker_hosts[2:]
    else:
        tf_config['cluster']['worker'] = worker_hosts[1:]

    if task_type == "ps":
        tf_config['task']['type'] = 'ps'
        tf_config['task']['index'] = task_index
    if task_type == 'worker' and task_index == 0:
        tf_config['task']['type'] = 'chief'
        tf_config['task']['index'] = 0
    if task_type == 'worker' and task_index > 0:
        if use_evaluator:
            if task_index == 1:
                tf_config['task']['type'] = 'evaluator'
                tf_config['task']['index'] = 0
            else:
                tf_config['task']['type'] = 'worker'
                tf_config['task']['index'] = task_index - 2
        else:
            tf_config['task']['type'] = 'worker'
            tf_config['task']['index'] = task_index - 1

    os.environ['TF_CONFIG'] = json.dumps(tf_config)


def generator(data_dir, need_sep=False):
    data_dir = tf.compat.as_str(data_dir)
    print('[data dir:{}]'.format(data_dir))
    files = tf.io.gfile.glob(data_dir)
    files_sep = [file for file in files if
                 int(re.findall(r'part-(.+?)-', file)[0]) % NUM_WORKERS == FLAGS.task_index] if need_sep else files
    print('[total file size={}, need_sep={}]'.format(len(files_sep), need_sep))
    for file in files_sep:
        for line in tf.io.gfile.GFile(file, 'r'):
            arr = line.strip().split('\t')
            if len(column_names) == len(arr):
                item = {}
                for k, v in zip(column_names, arr):
                    item[k] = v
                label = item.pop('label')
                yield item, label


def main(argv):
    model_path = os.path.join(model_dir, FLAGS.version)
    print('[model path={}]'.format(model_path))

    gen_tf_config()

    if FLAGS.mode == 'train':
        train(model_path)
    elif FLAGS.mode == 'predict':
        day = datetime.datetime.now().strftime('%Y-%m-%d') if FLAGS.day == '1' else FLAGS.day
        result_path = os.path.join(result_dir, 'day={}'.format(day))
        predict(model_path, FLAGS.ckpt_version, result_path)
    else:
        print('mode not in [train/eval/predict]')


if __name__ == '__main__':
    app.run(main)
