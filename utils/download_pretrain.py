import os
import urllib.request


def report(block_count, block_size, content_size):
    if block_count % (content_size // block_size // 5) == 1:
        print("Downloaded %.1f/100" % (block_size * block_count / content_size * 100))


def download(prefix, epoch):
    dir_name = os.path.dirname(prefix)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    base_name = prefix.replace("pretrain_model/", "") + "-%04d.params" % epoch
    save_name = "%s-%04d.params" % (prefix, epoch)
    base_url = os.environ.get("SIMPLEDET_BASE_URL", "https://1dv.alarge.space/")
    full_url = base_url + base_name

    try:
        print("Downloading %s from %s" % (save_name, full_url))
        urllib.request.urlretrieve(full_url, save_name, report)
    except Exception as e:
        print("Fail to download %s. You can mannually download it from %s and put it to %s" % (base_name, full_url, save_name))
        os.remove(save_name)
        raise e
