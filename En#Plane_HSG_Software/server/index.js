const express = require("express");
const app = express();
var http = require('http');
var path = require('path');
var static = require('serve-static');
const bodyParser = require("body-parser");
var requestIp = require('request-ip');
var moment = require('moment');
require('moment-timezone');
moment.tz.setDefault("Asia/Seoul");

app.use(static(path.join(__dirname,'/')));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

const port = process.env.PORT || 3000
const { PythonShell } = require("python-shell");

//let pyshell = new PythonShell('./hs_detection/YoutubePredict.py');
app.get('/', (req, res) => {
    //res.send("<h1>안녕하세유 모은서버에유 ^ㅁ^</h1>");
    res.sendFile(path.join(__dirname + '/test.html'));
    console.log("들어왔쥬!"); //index.html 파일은 무시하면 됨
    console.log("client IP: " +requestIp.getClientIp(req));


})
//라우터 정리 필요
app.post('/api/video', (req, res) => {
//    res.send('hello world')
var receivedUrl = req.body.url;
console.log("/api/video")
console.log("client IP: " +requestIp.getClientIp(req));
console.log("currentUrl : ",receivedUrl)
var startTime = moment().format('YYYY-MM-DD HH:mm:ss');
console.log("StartTime :", startTime);

//var url = receivedUrl;
var url = 'https://www.youtube.com/watch?v=2VZc0XaTGqM'

let options = {
    scriptPath: './hs_detection',
    args: [url]
};



PythonShell.run("YoutubePredict.py", options, function (err, data) {
    if (err) throw err;
    console.log("python에서 받은 data", data);

    //console.log(data[0])
    
    if(data) { res.json(data)}
    var endTime = moment().format('YYYY-MM-DD HH:mm:ss');
    console.log("EndTime :", endTime);
});
let data =  [ '66.69~89.369', '75.0~82.619', '79.11~83.64', '83.64~89.369' ];

// let data =  [ '66.69~89.369' ];
// res.status(200).json(data)
})


app.post('/api/text', (req, res) => {
    console.log("/api/text")
    console.log("client IP: " +requestIp.getClientIp(req));
    startTime = moment().format('YYYY-MM-DD HH:mm:ss');
    console.log("StartTime :", startTime);
    var receivedText = req.body.text;
    // console.log("currentText : ", receivedText)
    // receivedText = new String(receivedText);
    // console.log(receivedText);
    // if(receivedText.startsWith('“Fuck'))
    // {
    //     var text = "**** you, **** go back to Africa. The slave ship is loading up,” he said. Then he added an exclamation point: “Trump!"
    //     res.json((text)) 
    // }
    // else if(receivedText.startsWith('By March,')){
    //     text = "By March, a black woman in Houston reported that she was told by a white man that Trump was going to “*** ** *****.. get rid of all you ******.”"
    //     res.json((text)) 
    // }
    // else{
    //     return;
    // }
    let options = {
        mode:'text',
        pythonOptions: ['-u'],
        scriptPath: './hs_detection',
        encoding : 'utf8',
        args: [receivedText]
    };

    PythonShell.run("PredictText.py", options, function (err, results) {
        if (err) throw err;
        console.log("server: ",results);
        let data = results[0].replace(`b\'`, '').replace(`\'`, '');
        let buff = Buffer.from(data, 'base64'); 
        let text = buff.toString('utf-8');

        console.log("python에서 받은 data", (text));
        if (data) { 
            res.json((text)) 
            endTime = moment().format('YYYY-MM-DD HH:mm:ss');
            console.log("EndTime :", endTime);
        }
        
    });

    // let data =  [ '66.69~71.16', '75.0~82.619', '79.11~83.64', '83.64~89.369' ];
    //res.json(data)

})

app.listen(port, () => {
    console.log(`Server Running at ${port}`)
});


