 async function runExample(){
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

	//await myOnnxSession.loadModel("");
	const url = chrome.runtime.getURL('./data/cnn_model_v9.onnx');

	alert(url);

	//const outputMap = await myOnnxSession.run(inferenceInputs);
	//const outputTensor = outputMap.values().next().value;
	// chrome.storage.local.set({model: myOnnxSession}, () => {
	// console.log("Model is set");
	// });
}

runExample();