from prediction import network as net

dense_model = net.Model("dense")
trans_model = net.Model("transformer")

dense_opt = dense_model.init_conv_option
trans_opt = trans_model.init_transformer_option

f = 128
n = (3, 6, 3)

dense_net = lambda f=f, n=n: (dense_model.init_conv_option(n, f), dense_model)
se_dense_net = lambda f=f, n=n: (dense_model.init_conv_option(n, f, se=True), dense_model)
cbam_dense_net = lambda f=f, n=n: (dense_model.init_conv_option(n, f, cbam=True), dense_model)
dense_next = lambda f=f, n=n: (dense_model.init_conv_option(n, f, dense_next=True), dense_model)
erase_dense_net = lambda f=f, n=n: (dense_model.init_conv_option(n, f, erase_relu=True), dense_model)
bot_dense_net = lambda f=f, n=n: (dense_model.init_conv_option(n, f, bot=True), dense_model)
sam_se_lambda_bot_dense_next = \
    lambda f=f, n=n: (dense_opt(n, f, lambda_bot=True, sam=True, se=True, dense_next=True), dense_model)
sam_dense_net = lambda f=f, n=n: (dense_model.init_conv_option(n, f, sam=True), dense_model)
sam_dense_next = lambda f=f, n=n: (dense_opt(n, f, sam=True, dense_next=True), dense_model)
sam_se_dense_next = lambda f=f, n=n: (dense_opt(n, f, sam=True, dense_next=True, se=True), dense_model)
sam_se_bot_dense_next = lambda f=f, n=n: (dense_opt(n, f, sam=True, dense_next=True, bot=True, se=True), dense_model)
sam_se_vit_dense_next = lambda f=f, n=n: (dense_opt(n, f, sam=True, dense_next=True, se=True, vit=True), dense_model)

mix_net = lambda f=f, n=n: (dense_model.init_conv_option(n, f, mix_net=True), dense_model)
sam_mix_next = lambda f=f, n=n: (dense_opt(n, f, mix_net=True, sam=True, dense_next=True), dense_model)

lambda_net = lambda f=f, n=n: (dense_opt(n, f, lambda_net=True), dense_model)
sam_lambda_net = lambda f=f, n=n: (dense_opt(n, f, lambda_net=True, sam=True), dense_model)
sam_se_lambda_net = lambda f=f, n=n: (dense_opt(n, f, lambda_net=True, sam=True, se=True), dense_model)

pyconv = lambda f=f, n=n: (dense_opt(n, f, pyconv=True), dense_model)
sam_pyconv = lambda f=f, n=n: (dense_opt(n, f, pyconv=True, sam=True), dense_model)
sam_se_pyconv = lambda f=f, n=n: (dense_opt(n, f, pyconv=True, sam=True, se=True), dense_model)

sam_vit = lambda f=128, n=(12,): (trans_opt(n, f, 14, 8, sam=True), trans_model)
sam_lambda_transformer = lambda f=128, n=(12,): (trans_opt(n, f, 14, 4, sam=True, lambda_=True), trans_model)
sam_cvt = lambda f=128, n=(3, 6, 3): (trans_opt(n, f, 4, 8, sam=True), trans_model)

def build_model(model, input_shape: tuple, output_size: int):
    model = model()[-1].build_model(input_shape, output_size, )

    return model
