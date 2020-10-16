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
		const user_url = "https://www.youtube.com"+$(curr).parent().parent().prev().find("a#author-text").attr('href');
		//console.log(user_url);
		const response = await fetch('http://localhost:8080/get_tokens', {
			method: 'POST',
			body: JSON.stringify({'text': text, 'user_url': user_url}),
			mode: 'cors',
			headers: {
		  		'Content-Type': 'application/json'
			},
		});
		const data = await response.json();
		let color = "green";
		let span_text = ' Бот на '+'0%';
		const metric_span = document.createElement('span');
		//console.log(data);
		if(data['is_in_bots']){
			color = "red";
			span_text = " Найден в БД";
		}else{
			const inferenceInputs =  [
				new Tensor(new Int32Array (data['tokens']), "int32", [1, 748]),
			];

			const outputMap = await session.run(inferenceInputs);
		    const outputTensor = outputMap.values().next().value;
		    const bot_percentage = (outputTensor.data[1]*100).toFixed(2);
		    if(bot_percentage > 50){
		    	color = 'orange';
		    	span_text = ' Возможно бот!';
		    }else{
		    	span_text = '';
		    }
		    
		}

	    metric_span.innerHTML = span_text;
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