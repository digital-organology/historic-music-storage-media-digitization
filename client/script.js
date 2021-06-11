// Creates canvas 320 × 200 at 10, 50
var paper = Raphael(0, 0, 4000, 3500);


fetch("test_data.json").then(res => res.json()).then(data => {
    arr = data["data"];

 
    for (var obj of arr) {

        var id = obj["id"];
        console.log(id);
        var track = obj["track"];
        var points = obj["points"];

        points.forEach(function (point, idx) {

            var rect = paper.rect(point[0], point[1], 0.001, 0.001);
            rect.attr("stroke", "green");
        });
    };
});