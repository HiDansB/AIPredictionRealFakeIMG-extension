// Kiểm tra model.onnx có tồn tại không
fetch(chrome.runtime.getURL('model/model.onnx'), { method: 'HEAD' })
  .then(r => {
    const dot  = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    if (r.ok) {
      dot.className    = 'dot ok';
      text.textContent = '✓ Model sẵn sàng · Có thể phân tích ảnh';
    } else {
      dot.className    = 'dot warn';
      text.textContent = '⚠ Thiếu model.onnx — đặt vào extension/model/';
    }
  })
  .catch(() => {
    document.getElementById('statusDot').className    = 'dot warn';
    document.getElementById('statusText').textContent = '⚠ Thiếu model.onnx';
  });
