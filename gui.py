# coding=utf-8
import wx


class Example(wx.Frame):
    def __init__(self, parent, title, size):
        super(Example, self).__init__(parent, title=title, size=size)
        self.panel = wx.Panel(self)
        self.tranningData_1 = wx.TextCtrl(self.panel)  # 训练数据
        self.tranningData_2 = wx.TextCtrl(self.panel)
        self.path = wx.TextCtrl(self.panel)  # 文件保存路径
        self.InitUI()
        self.Centre()
        self.Show()

    def InitUI(self):
        sizer = wx.GridBagSizer(0, 0)
        sizer.Add(wx.StaticText(self.panel, label="训练数据范围"), pos=(1, 0), flag=wx.ALL, border=5)
        sizer.Add(self.tranningData_1, pos=(2, 0), flag=wx.EXPAND | wx.ALL, border=5)
        sizer.Add(self.tranningData_2, pos=(2, 1), flag=wx.EXPAND | wx.ALL, border=5)
        sizer.Add(wx.StaticText(self.panel, label="评估结果保存路径"), pos=(3, 0), flag=wx.ALL, border=5)
        sizer.Add(self.path, pos=(4, 0), flag=wx.ALL, border=5)
        button = wx.Button(self.panel, label="开始评估")
        sizer.Add(button, pos=(4, 1), flag=wx.ALL, border=5)
        self.panel.SetSizerAndFit(sizer)


app = wx.App()
Example(None, title='YiGuTong', size=(270, 220))
app.MainLoop()
