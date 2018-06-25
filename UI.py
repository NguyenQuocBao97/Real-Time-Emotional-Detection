import wx
FRAME_SIZE = (480,600)
BUTTON_SIZE = (200,50)
CANCEL_BTN_POS = (25,500)
START_BTN_POS = (250,500)
IMAGE_POS = (25,25) 
class HelloFrame(wx.Frame):
    def __init__(self,*args,**kw):
        wx.Frame.__init__(self,None,-1,'Emotion Detector',size=FRAME_SIZE, style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
        pnl = wx.Panel(self)
        png = wx.Bitmap("C:\Users\ACER\Desktop\Real-Time-Emotional-Detection-master\Danh.png", wx.BITMAP_TYPE_ANY)
        wx.StaticBitmap(pnl, -1, png, IMAGE_POS)
        startButton = wx.Button(pnl, label='Start Emotion Detecting',pos=START_BTN_POS,size=BUTTON_SIZE)
        startButton.SetDefault()
        cancelButton = wx.Button(pnl, label='Cancel',pos=CANCEL_BTN_POS,size=BUTTON_SIZE)
        btnSizer = wx.StdDialogButtonSizer()
        btnSizer.AddButton(startButton)
        btnSizer.AddButton(cancelButton)
        self.Bind(wx.EVT_BUTTON, self.closePressed,cancelButton)
        self.Bind(wx.EVT_BUTTON, self.startPressed,startButton)
        btnSizer.Realize()
    def closePressed(self,event):
        self.Close(True)
    def startPressed(self,event):
        self.Close(True)