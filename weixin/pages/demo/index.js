//获取应用实例
const app = getApp();

const loaded_data = require('../../hamlet.js');
const asr_several_url = 'http://www.enginewang.fun:4000/data'
const resource_url = "http://www.enginewang.fun/~lls/resource/"

function parseURL(s) {
  // console.log(s)
  return resource_url + s
}

const button_images = {
  record: [parseURL('image/button1.jpg'), parseURL('image/button2.png')],
  next: parseURL('image/button_next.jpg'),
  reset: parseURL('image/button_reset.jpeg'),
}

Page({
  /**
   * 页面的初始数据
   */
  data: {
    bg_image: '',

    button_record_bg: button_images.record[0],
    button_record_display: 'none',
    button_next_bg: button_images.next, 
    button_next_display: 'inline',
    button_reset_bg: button_images.reset, 
    button_reset_display: 'inline',

    story: {},
    story_content: 'None',
    story_answers: [
      { answer: 'aaa', index: 'A', next: 1, position_top: "70%", color: "white"}, 
      { answer: 'bbb', index: 'B', next: 2, position_top: "75%", color: "white"}
      ],
    story_id: -1,

    record_state: 0,
    record_file: '',

    asr_hypos: '',
    answer_index: '',
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    // init story content
    console.log(loaded_data.story)
    var story = {}
    for (var x of loaded_data.story) {
      story[x.id] = x
    }
    this.setData({ story: story })
    console.log(this.data.story)
    this.updateStoryContent()

    // init recoder
    var that = this
  
    this.recorderManager = wx.getRecorderManager();
    this.recorderManager.onError(function () {
      console.log('[ERROR] [录音失败]！')
    });
    this.recorderManager.onStop(function (res) {
      that.setData({
        record_file: res.tempFilePath
      })
      console.log('[录音完成]')
      console.log(res.tempFilePath)
      console.log(that.data.story_answers)

      // play the audio
      that.innerAudioContext.src = res.tempFilePath
      that.innerAudioContext.play()

      // ASR recognition
      that.sendDataToServer(res.tempFilePath)
    });

    this.innerAudioContext = wx.createInnerAudioContext();
    this.innerAudioContext.onError((res) => {
      console.log("[ERROR] [播放录音失败]！")
    })
  },

  /**
   * 触击“欢迎页面”直接重定向到首页
   */
  goHome: function (e) {
    //清空计时器
    clearInterval(this.data.timerId);
    //关闭当前页，重定向到首页
    wx.switchTab({
      url: '../index/index'
    })
  },

  sendDataToServer: function (filepath) {
    var that = this

    console.log('[sendDataToServer] ')
    console.log('>>host = ' + asr_several_url)
    console.log('>>filepath = ' + filepath)

    // get selector
    var selector = {}
    for (var x of this.data.story_answers) {
      selector[x.index] = x.answer
    }
    console.log('>>send selector = ' + JSON.stringify(selector))

    // send file and selector
    wx.uploadFile({
      url: asr_several_url,
      filePath: filepath, //文件临时路径
      name: 'audio',
      // header: {
      //   "Content-Type": "multipart/form-data"
      // },
      formData: { selector: JSON.stringify(selector) },
      success: function (res) {
        console.log('>>response = ' + res.data)
        var res_data = JSON.parse(res.data)
        const answer_index = res_data.tag
        const answer_str = res_data.str

        console.log('>>answer_index = ' + answer_index)
        console.log('>>answer_str = ' + answer_str)
        // store the ASR results
        that.setData({ 
          asr_hypos: answer_str, 
          answer_index: answer_index
          })
        that.updateStoryContent()
      },
      fail: function (res) {
        that.tip('上传文件失败，使用建议分支:\n' + res.data)
        that.setData({
          asr_hypos: '这里应该显示识别的结果',
          answer_index: 'A',
        })
        that.updateStoryContent()
      },
      complete: function (res) {
      }
    })
  },

  onRecord: function () {
    var that = this

    if (this.data.record_state == 0) {
      this.recorderManager.start({
        duration: 60000,
        sampleRate: 16000,
        numberOfChannels: 1,
        encodeBitRate: 48000,
        format: 'mp3',
        //frameSize: 50
      });
      this.setData({ record_state: 1 })
    }
    else {
      this.recorderManager.stop()
      this.setData({ record_state: 0 })
    }

    this.setData({ button_record_bg: button_images.record[this.data.record_state]})
  },

  onNext: function() {
    // next page
    this.updateStoryId(this.data.answer_index)
    this.updateStoryContent()
  },

  onReset: function() {
    this.setData({
      story_id: -1,
      asr_hypos: "",
      answer_index: "",
    })
    this.updateStoryContent()
  },

  updateStoryContent: function () {
    var data = this.data.story[this.data.story_id]

    // console.log(data)
    var cur_answers = new Array()
    for (var i=0; i<data.answers.length; i++) {
      cur_answers[i] = {}
      Object.assign(cur_answers[i], data.answers[i])
      cur_answers[i].position_top = (70 + 10 * i).toString() + "%"
      if (data.answers[i].index == this.data.answer_index) {
        cur_answers[i].color = "red"
        cur_answers[i].answer += '\n[' + this.data.asr_hypos + ']'
      } else {
        cur_answers[i].color = "white"
      }
    }

    this.setData({
      bg_image: parseURL(data.image),
      story_content: data.content,
      story_answers: cur_answers
    })

    if (data.answers.length == 0) {
      this.setData({button_record_display: "none"})
    }
    else {
      this.setData({ button_record_display: "inline" })
    }

  },

  updateStoryId: function(answer_index) {
    var data = this.data.story[this.data.story_id]
    var story_id = this.data.story_id

    if (data.answers.length == 0) {
      story_id = story_id + 1
    } else {
      story_id = -1
      for (var i = 0; i < data.answers.length; i++) {
        if (data.answers[i].index == answer_index)
          story_id = data.answers[i].next
      }
      if (story_id == -1) {
        this.tip('请录音！')
        return
      }
    }

    this.setData({
      story_id: story_id,
      asr_hypos: "",
      answer_index: "",
      })
    console.log("[updateStoryId] update story-id = " + story_id)
  },

  tip: function (msg) {
    wx.showModal({
      title: '提示',
      content: msg,
      showCancel: false
    })
  }

})