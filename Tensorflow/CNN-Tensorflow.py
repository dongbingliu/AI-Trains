# coding=utf-8
# 下载数据
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

#在jupyter notebook中显示图片片
import matplotlib.pyplot as plt
img = mnist.train.images[0].reshape(28,28)
print (img)
plt.imshow(img, cmap='gray')
plt.show()
# input placeholders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
# img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])
#
kernel1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# Conv -> (?, 28, 28, 32)
# Pool -> (?, 14, 14, 32)
# 卷积层,调用用tf.nn.conv2d()
L1 = tf.nn.conv2d(X_img, kernel1, strides=[1, 1, 1, 1], padding='SAME')
# 用用3*3的卷积核,1个通道,使用用32个卷积核;(有多少个卷积核,就有多少个输出通道
# 1个batch,步⻓长1*1,1个通道;
# padding采用用边缘补零
# 使用用relu激活函数,调用用tf.nn.relu()
L1 = tf.nn.relu(L1)
# 最大大池化层,调用用tf.nn.max_pool()
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
strides=[1, 2, 2, 1], padding='SAME')

kernel2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
# 用用3*3的卷积核,32个通道,使用用64个卷积核;
# 1个batch,步⻓长1*1,1个通道;
# padding采用用边缘补零
L2 = tf.nn.conv2d(L1, kernel2, strides=[1, 1, 1, 1], padding='SAME')
# L2 shape = [?, 14, 14, 64]
L2 = tf.nn.relu(L2)
# L2 shape = [?, 7, 7, 64]
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
padding='SAME')
# L2 shape = [?, 7, 7, 64],?表示图片片的数量量
L2 = tf.reshape(L2, [-1, 7 * 7 * 64])
# [-1, 7 * 7 * 64],-1表示图片片的数量量,我们不不用用指定,tf.reshape()函数自自动计算;把L2

# Final FC 7x7x64 inputs -> 10 outputs
W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 10], stddev=0.01))
b = tf.Variable(tf.random_normal([10]))
# y = W3*L2 + b
# 乘法使用用的是 tf.matmul 函数)
hypothesis = tf.matmul(L2, W3) + b

cost =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,
labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


# 创建对话
sess = tf.Session()
# 变量量初始化
sess.run(tf.global_variables_initializer())
# training_epochs:1个epoch表示全部图片片训练一一次
training_epochs = 1
# batch_size:1个batch表示计算一一次cost使用用的图片片数量量
batch_size = 32
# 训练模型
print('Learning stared. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _, = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
        print('Epoch:', '%04d' % (epoch + 1), 'cost =',
              '{:.9f}'.format(avg_cost))


print('Learning Finished!')
print ("预测数据")


# 测试模型和计算准确率
# tf.argmax函数可以在一一个张量量里里里沿着某条轴的最高高条目目的索引值
# tf.argmax(y,1) 是模型认为每个输入入最有可能对应的那些标签
# 而而 tf.argmax(y_,1) 代表正确的标签,我们可以用用 tf.equal 来检测我们的预测是否真实标签匹配

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# correct_prediction会得到一一组布尔值。
# 为了了确定正确预测项的比比例例,我们可以把布尔值转换成浮点数,然后取平均值。
# 例例如, [True, False, True, True] 会变成 [1,0,1,1] ,取平均值后得到 0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 我们计算所学习到的模型在测试数据集上面面的正确率
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y:
mnist.test.labels}))