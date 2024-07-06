# Tinder-Auto-Liker-and-Disliker

我研究了Tinder的API，發現Tinder的API其實是開放的，這代表什麼呢?這代表你不需要使用Tinder這個APP，也能夠實現Tinder上面的左滑右滑，滑自己喜歡的人。

我的作法是這樣的，首先網路上面有Tinder公開的API，可以在Python上面使用，我先pip install，然後接著就是取得Tinder每個帳號的auth token，你只要登入Tinder的話就會有這個，這個是用來識別你是誰的。
只要有了這些基本上就可以做到任何Tinder可以做到的事情。

至於怎麼挑選喜歡的女生呢?我這邊是使用卷積神經網路ResNet-50去訓練的，我將ResNet-50的資料集分成兩個class，一個叫做「woman」，另一個叫做「not_woman」。我在woman類別裡面，放了一大堆喜歡的女生的類型的照片，但由於照片可能太少，所以我還要進行Data Augmentation，把照片的數量增加至三倍，同時也可以增加照片的多樣性。然後我發現，Tinder上面很多人除了放自己的照片外，還會放風景、貓狗、食物等等，所以我在not_woman這邊放了各種風景、貓狗跟食物的照片，並且再進行Data Augmentation。這樣就可以分辨出哪些照片是女生哪些不是。
接著就是訓練這個AI Model，我訓練的epoch偏少，大約才17左右，但是有成功讓loss function的值下降。
訓練完之後，可以試著去跑ResNet-50的模型，這是一個關於圖像識別的AI，模型會回傳兩個值，值的數量是跟你的類別數量是一樣的，分別是woman跟not_woman的相似度，也就是說，如果這張圖越像我喜歡的女生，那woman回傳的值就會越高，如果越不像，那就越低。而且如果AI認為他是一個woman，那not_woman的值一定會是負的。

最後，我的自動左右滑Tinder的程式的程序是這樣的。

1. 先登入去找我的auth token。
2. 將auth token輸入至我的python程式，並且讓他登入後可以去尋找配對對象。
3. 載入我訓練好的AI Model。
4. 將配對對象的照片下載下來並且放入AI Model進行分辨。
5. 找出該帳戶所有屬於女生的照片，並且依據AI給的相似度取平均值(如果不是女生的照片就跳過)。
6. 如果女生的照片不超過兩張或是分數過低，左滑，反之，右滑。
7. 繼續找下一個配對對象。

我的AI Model的下載網址在這，由於檔案過大，所以放在雲端: https://ttuedutw-my.sharepoint.com/:u:/g/personal/410806228_o365_ttu_edu_tw/Eehh-NZvRvhInWdOgulHO3MBfMxQkKfB3Df2CJIcVcqD7w?e=fiAfge
