async function get_session(){
     // console.log(URL.createObjectURL("./cnn_model.onnx"));
    console.log("Run");
     

	const myOnnxSession = new onnx.InferenceSession();

	      // load the ONNX model file
	      // const response = await fetch('http://localhost:8080/get_tokens', {
	      //   method: 'POST',
	      //   body: JSON.stringify({'text': 'Дима молодец красава!'}),
	      //   mode: 'cors',
	      //   headers: {
	      //     'Content-Type': 'application/json'
	      //   },
	      // });
	      // const tokens = await response.json();
	      // console.log(tokens['tokens']);
	      // const inferenceInputs =  [
	      //     new Tensor(new Int32Array (tokens['tokens']), "int32", [1, 748]),
	      // ];

	const id = chrome.runtime.id;

	await myOnnxSession.loadModel("chrome-extension://"+id+"/data/cnn_model_v9.onnx");
	//const outputMap = await myOnnxSession.run(inferenceInputs);
	//const outputTensor = outputMap.values().next().value;
	return myOnnxSession;
}

async function main(){

	const session = await get_session();
	let checked = -1;
	async function detect(curr, index){

		if(index <= checked){
			return;
		}
		checked = index;
		const text = $(curr).text();
		const response = await fetch('http://localhost:8080/get_tokens', {
			method: 'POST',
			body: JSON.stringify({'text': text}),
			mode: 'cors',
			headers: {
		  		'Content-Type': 'application/json'
			},
		});
		const tokens = await response.json();

		const inferenceInputs =  [
			new Tensor(new Int32Array (tokens['tokens']), "int32", [1, 748]),
		];

		const outputMap = await session.run(inferenceInputs);
	    const outputTensor = outputMap.values().next().value;
	    const bot_percentage = (outputTensor.data[1]*100).toFixed(2);
	    let color = 'green';
	    if(bot_percentage > 50){
	    	color = 'red';
	    }
	    const metric_span = document.createElement('span');
	    metric_span.innerHTML = ' Бот на '+bot_percentage+'%';
	    $(metric_span).css({color: color});
	    $(curr).parent().parent().prev().find("yt-formatted-string a").append(metric_span);

	};


	setInterval(() => {
		const comments = $("ytd-comments#comments ytd-comment-thread-renderer yt-formatted-string#content-text");
		for (let i = 0; i < comments.length; i++) {
			detect(comments[i], i);
		}
	}, 5000);
}

main();
//$(document).ready();