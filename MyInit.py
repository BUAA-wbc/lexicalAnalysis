import os
import tempfile
import time
import logging
import sys
import marshal
from hashlib import md5
from math import log
import re

DEFAULT_DICT=None
DEFAULT_DICT_NAME="dict.txt"

re_eng=re.compile("a-zA-Z0-9",re.U)

log_console=logging.StreamHandler(sys.stderr)#sys.stderr错误输出
default_logger=logging.getLogger(__name__)#使用getLogger()对象后,为他设置级别,创建logger对象
default_logger.setLevel(logging.DEBUG) #设置日志级别
default_logger.addHandler(log_console) #添加日志处理器,定义日志处理器让其输出到指定地方

_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))


class Tokenizer(object):
    def __init__(self,dictionary=DEFAULT_DICT):
        if dictionary==DEFAULT_DICT:
            self.dictionary=dictionary
        else:
            self.dictionary=_get_abs_path(dictionary)
        self.total=0
        self.initialized=False
        self.cache_file=False
        self.tmp_dir = None
    #读取词频文件
    def gen_pfdict(self,f_name):
        lfreq={}
        ltotal=0
        with open(f_name,'rb') as f:
            for lineno,line in enumerate(f,1):
                try:
                    line=line.strip().decode("utf-8")
                    word,freq=line.split(' ')[:2]
                    freq=int(freq)
                    lfreq[word]=freq
                    ltotal+=freq
                    for ch in range(len(word)):#for example:word="龟龙片甲",wfrag=龟,鬼龙,鬼龙片,鬼龙片甲 全部加入词典
                        wfrag=word[:ch+1]
                        if wfrag not in lfreq:
                            lfreq[wfrag]=0
                except ValueError:
                    raise ValueError("invalid dictionary entry in %s at Line %s:%s" % (f_name,lineno,line))
        f.close()
        return lfreq,ltotal


    #词频文件的初始化
    def initialize(self,dictionary=None):
        if dictionary: #如果dictionary不为空
            abs_path=_get_abs_path(dictionary)
            if self.dictionary==abs_path and self.initialized:
                return
            else:
                self.dictionary=abs_path
                self.initialized=False
        else:
            abs_path=self.dictionary

        t1=time.time()
        if self.cache_file:
            cache_file=self.cache_file
        elif abs_path==DEFAULT_DICT:
            cache_file="jieba.cache"
        else:
            cache_file="jieba.u%s.cache"%md5(abs_path.encode('utf-8','replace')).hexdigest()
        cache_file=os.path.join(self.tmp_dir or tempfile.gettempdir(),cache_file)
        tmpdir=os.path.dirname(cache_file)
        load_from_cache_fail=True
        # isfile() isdir()判断是否是文件和目录 getmtime() 获取文件的创建时间
        if os.path.isfile(cache_file) and (
                abs_path == DEFAULT_DICT or os.path.getmtime(cache_file) > os.path.getmtime(abs_path)):
            default_logger.debug("Logging model from cache %s" % cache_file)
            try:
                with open(cache_file, "rb") as cf:
                    self.FREQ, self.total = marshal.load(cf)  # marshal.load() 将二进制数据反序列为python对象
                load_from_cache_fail = False
            except Exception:
                load_from_cache_fail = True
        if load_from_cache_fail:
            self.FREQ,self.total=self.gen_pfdict(abs_path)
            try:
                with open(cache_file,'wb') as temp_cache_file:
                    marshal.dump((self.FREQ,self.total),temp_cache_file)
            except Exception:
                default_logger.exception("Dump cache file failed.")
        self.initialized=True
        default_logger.debug("Loading model cost %0.3f seconds." % (time.time()-t1))
        default_logger.debug("Prefix dict has been built successfully.")

    #检查词频文件是否初始化
    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def cut(self,sentence,cut_all=False,HMM=True):
        self.check_initialized()
        DAG=self.getDAG(sentence)
        route={}
        self.calc(sentence,DAG,route)


    #获取有向无环图
    def getDAG(self,sentence):
        self.check_initialized()
        DAG={}   #DAG空字典,用来构建DAG的有向无环图
        N=len(sentence)
        for k in range(N):
            tmplist=[]  #从字开始能在FREQ中的匹配到词的末尾所在的list
            i=k
            frag=sentence[k] #取传入词中的值  sentence="去北京大学玩",frag=去 北 京 大 学 玩 ,frag是里面的每一个字
            while i<N and frag in self.FREQ:   #当传入的词,在FREQ中,就给tmplist赋值,构建字开始可能去往的所有路径
                # if frag in self.FREQ:
                #     print(frag+ " in self.FREQ")
                # print(frag+" : "+str(self.FREQ[frag]))
                if self.FREQ[frag]:  # 每个字,在FREQ中进行查找,如果查到,就将下标传入tmplist中
                    tmplist.append(i)  #添加词语所在的位置
                i+=1                   #查找"去",后继续查找"去北"是否也在语料库中,直到查不到退出循环
                frag=sentence[k:i+1]
            if not tmplist:   #如果tmplist为空的话,也就是某个字没有在词典中,那么将这个字单独加入tmplist
                tmplist.append(k)
            DAG[k]=tmplist
        return DAG

    #动态规划查找最大概率路径 也即是计算最大概率的切分组合
    def calc(self,sentence,DAG,route):
        N=len(sentence)
        route[N]=(0,0)
        logtotal=log(self.total)    #对概率值取对数之后的结果 total是词表中共有多少词,共有60101967个词,log使概率相乘的计算变成对数相加,防止溢出
        for idx in range(N-1,-1,-1): #range()从N-1到0,从后往前遍历句子,反向计算最大概率
            # route[idx]=max((log(self.FREQ.get(sentence[idx:x+1])or 1)-logtotal+route[x+1][0],x)for x in DAG[idx])
            #log(self.FREQ.get(sentence[idx:x+1]) or 1) 如果词语在词典中出现,则返回频率,否则,返回1,计算对数
            #log(xxx)-logtotal+route[x+1][0]
            #max((4,x)for x in range(5))会返回(4,4),返回(4,(0-5之间的最大值))

            maxNum=0
            for x in DAG[idx]:
                maxNum=max(maxNum,x)
            route[idx]=(log(self.FREQ.get(sentence[idx:x+1])or 1)-logtotal+route[x+1][0],maxNum)
            # print("+++++++++++++++++++++++++++++++++++++++++")
            # print("idx:"+str(idx))
            # print("maxNum:"+str(maxNum))
            # print(log(self.FREQ.get(sentence[idx:x+1])or 1)-logtotal)
            # print("x:"+str(x))
            # print(route[x+1][0])
            # print("+++++++++++++++++++++++++++++++++++++++++")
        # print(route)
        #列表推倒求最大概率对数路径
        #route[idx]=max([(概率对数,词语末字位置) for idx in DAG[idx]])
        #以idx:(概率对数最大值,词语末字位置)键值对的形式保存在route中
        #route[x+1][0]表示词路径[x+1,N-1]的最大概率对数
        #[x+1][0]即表示去句子x+1位置对应元组(概率对数,词语末字位置)的概率对数



    def get_word_freq(self,word):
        if word in self.FREQ:
            return self.FREQ[word]
        else:
            return 0

    def cut_DAG_NO_HMM(self,sentence):
        self.check_initialized()
        DAG = self.getDAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)
        x=0
        N=len(sentence)
        buf=""
        while x<N:
            y=route[x][1]+1
            l_word=sentence[x:y] #得到以x位置起点的最大概率切分词语
            print(y)
            print(l_word)
            if re_eng.match(l_word) and len(l_word)==1:
                buf+=l_word
                x=y
            else:
                if buf:
                    yield buf
                    buf=""
                yield l_word
                x=y
        if buf:
            yield buf
            buf=""

if __name__=='__main__':
    s=u"去北京大学玩"
    t=Tokenizer("dict.txt")
    dag=t.getDAG(s)

    print("/"+s+"/ 的前缀词典:")   #打印前缀词典的词频数
    for pos in range(len(s)):
        print(s[:pos+1],t.get_word_freq(s[:pos+1]))


    print(t.total)      #打印总词频数


    for d in dag:      #每个节点的有向无环图
        print(d,":",dag[d])


    route={}
    t.calc(s,dag,route)   #计算最大概率
    print("route:")
    print(route)


    # print("/".join(t.cut_DAG_NO_HMM(s)))
    print("/".join(t.cut_DAG_NO_HMM(s)))  #t.cut_DAG_NO_HMM(s)返回的是一个生成器