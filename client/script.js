$('#main-recent-filter').toggle();
$('#main-recent-filter').draggable();

// Creates canvas 320 × 200 at 10, 50
var paper = Raphael("container", 4500, 4500);
paper.setViewBox(0, 0);
// paper.setViewBox(0, 0, 5000, 5000, true);
// paper.setSize('70%', '70%');

// var selectionViewCoords = [];
var maxTrackId = 0;
/* List of useful colors
#8dd3c7
#ffffb3
#bebada
#fb8072
#80b1d3
#fdb462
#b3de69
#fccde5
#d9d9d9
#bc80bd
#ccebc5
#ffed6f */
const CSS_COLOR_NAMES = [
    "AliceBlue",
    "AntiqueWhite",
    "Aqua",
    "Aquamarine",
    "Azure",
    "Beige",
    "Bisque",
    "Black",
    "BlanchedAlmond",
    "Blue",
    "BlueViolet",
    "Brown",
    "BurlyWood",
    "CadetBlue",
    "Chartreuse",
    "Chocolate",
    "Coral",
    "CornflowerBlue",
    "Cornsilk",
    "Crimson",
    "Cyan",
    "DarkBlue",
    "DarkCyan",
    "DarkGoldenRod",
    "DarkGray",
    "DarkGrey",
    "DarkGreen",
    "DarkKhaki",
    "DarkMagenta",
    "DarkOliveGreen",
    "DarkOrange",
    "DarkOrchid",
    "DarkRed",
    "DarkSalmon",
    "DarkSeaGreen",
    "DarkSlateBlue",
    "DarkSlateGray",
    "DarkSlateGrey",
    "DarkTurquoise",
    "DarkViolet",
    "DeepPink",
    "DeepSkyBlue",
    "DimGray",
    "DimGrey",
    "DodgerBlue",
    "FireBrick",
    "FloralWhite",
    "ForestGreen",
    "Fuchsia",
    "Gainsboro",
    "GhostWhite",
    "Gold",
    "GoldenRod",
    "Gray",
    "Grey",
    "Green",
    "GreenYellow",
    "HoneyDew",
    "HotPink",
    "IndianRed",
    "Indigo",
    "Ivory",
    "Khaki",
    "Lavender",
    "LavenderBlush",
    "LawnGreen",
    "LemonChiffon",
    "LightBlue",
    "LightCoral",
    "LightCyan",
    "LightGoldenRodYellow",
    "LightGray",
    "LightGrey",
    "LightGreen",
    "LightPink",
    "LightSalmon",
    "LightSeaGreen",
    "LightSkyBlue",
    "LightSlateGray",
    "LightSlateGrey",
    "LightSteelBlue",
    "LightYellow",
    "Lime",
    "LimeGreen",
    "Linen",
    "Magenta",
    "Maroon",
    "MediumAquaMarine",
    "MediumBlue",
    "MediumOrchid",
    "MediumPurple",
    "MediumSeaGreen",
    "MediumSlateBlue",
    "MediumSpringGreen",
    "MediumTurquoise",
    "MediumVioletRed",
    "MidnightBlue",
    "MintCream",
    "MistyRose",
    "Moccasin",
    "NavajoWhite",
    "Navy",
    "OldLace",
    "Olive",
    "OliveDrab",
    "Orange",
    "OrangeRed",
    "Orchid",
    "PaleGoldenRod",
    "PaleGreen",
    "PaleTurquoise",
    "PaleVioletRed",
    "PapayaWhip",
    "PeachPuff",
    "Peru",
    "Pink",
    "Plum",
    "PowderBlue",
    "Purple",
    "RebeccaPurple",
    "Red",
    "RosyBrown",
    "RoyalBlue",
    "SaddleBrown",
    "Salmon",
    "SandyBrown",
    "SeaGreen",
    "SeaShell",
    "Sienna",
    "Silver",
    "SkyBlue",
    "SlateBlue",
    "SlateGray",
    "SlateGrey",
    "Snow",
    "SpringGreen",
    "SteelBlue",
    "Tan",
    "Teal",
    "Thistle",
    "Tomato",
    "Turquoise",
    "Violet",
    "Wheat",
    "White",
    "WhiteSmoke",
    "Yellow",
    "YellowGreen",
  ];

// initialise navigation position
var currentX = 0;
var currentY = 0;
var currentS = 4500;

$("#right").on("click",function(){
    currentX = currentX - 0.05 * currentX;
 paper.setViewBox(currentX, currentY, currentS, currentS, true);
   
});

$("#left").on("click",function(){
    currentX = currentX + 0.05 * currentX;
 paper.setViewBox(currentX, currentY, currentS, currentS, true);
   
   
});
$("#down").on("click",function(){
     currentY = currentY - 0.05 * currentY;
 paper.setViewBox(currentX, currentY, currentS, currentS, true);
});

$("#up").on("click",function(){
      currentY = currentY + 0.05 * currentY;
 paper.setViewBox(currentX, currentY, currentS, currentS, true);
   
});

$("#zoomout").on("click",function(){
      currentS = currentS + 0.1 * currentS;
 paper.setViewBox(currentX, currentY, currentS, currentS, true);
   
});
$("#zoomin").on("click",function(){
      currentS = currentS - 0.1 * currentS;
 paper.setViewBox(currentX, currentY, currentS, currentS, true);
   
});


$("#full").on("click",function(){
    currentX = 0;
    currentY = 0;
    currentS = 4500;
    paper.setViewBox(currentX, currentY, currentS, currentS, true);
});



function onNoteClick(e) {
    var trackId = e.target.dataset.track;
    var noteId  = e.target.id;
    var color   = CSS_COLOR_NAMES[trackId];

    $("#presentShapeValue").html(noteId);

    $("#presentTrackValue").html(trackId);
    $("#presentTrackColor").css("background-color", color);

    var trackUp   = trackId > 1           ? trackId-1 : trackId;
    var trackDown = trackId < maxTrackId  ? parseInt(trackId)+1 : trackId;

    $("#track1Value").html(trackUp);
    $("#track2Value").html(trackDown);

    $("#track1Color").css("background-color", CSS_COLOR_NAMES[trackUp]);
    $("#track2Color").css("background-color", CSS_COLOR_NAMES[trackDown]);

//    alert("Note ID: ".concat(e.target.id).concat("; Track ID: ").concat(e.target.dataset.track));
}

function hightlightNote(e) {
    track = e.target.dataset.track;
    trackElements = document.querySelectorAll('[data-track="' + track + '"]');
    trackElements.forEach(node => {
        node.classList.remove("inactive");
        node.classList.add("active");
    })
}

function lowlightNote(e) {
    track = e.target.dataset.track;
    trackElements = document.querySelectorAll('[data-track="' + track + '"]');
    trackElements.forEach(node => {
        node.classList.remove("active");
        node.classList.add("inactive");
    })
}

function onSaveClick(e) {
    alert("Save changes");
}

function onDiscardClick(e) {
    alert("Discard changes");
}




fetch("data.json")
    .then(res => res.json())
    .then(data => {
        data["data"].forEach(shape => {
            var note = paper.path(shape.points)
                            .attr({"fill" : CSS_COLOR_NAMES[shape.track]});
            node_id = "shape_".concat(shape.id);
            note.node.id = node_id;
            note.node.dataset.track = shape.track;
            note.node.dataset.points = shape.points;
            $("#".concat(node_id)).click(onNoteClick);
            $("#".concat(node_id)).on("mouseenter", hightlightNote);
            $("#".concat(node_id)).on("mouseout", lowlightNote);
            maxTrackId = Math.max(maxTrackId, shape.track);
        });
        // selectionViewCoords = data["center"];

    // $("body").append("<div id='selectionView' class='selectionView'></div>");
    $("#selectionView").css("top", 200);//selectionViewCoords[1]-selectionViewCoords[2]/2);
    $("#selectionView").css("left", 500);//, selectionViewCoords[0]-selectionViewCoords[2]/2);
    $("#selectionView").css("width", 200);//, selectionViewCoords[2]);
    $("#selectionView").css("height", 80);//, selectionViewCoords[2]);
    
    $("#track1Color").on("click", function(d) {
        var newTrack = $("#track1Value").html();
        var shapeId  = $("#presentShapeValue").html();
        setNewTrackForShape(shapeId, newTrack);

    });

    $("#track2Color").on("click", function(d) {
        var newTrack = $("#track2Value").html();
        var shapeId  = $("#presentShapeValue").html();
        setNewTrackForShape(shapeId, newTrack);

    });


    function setNewTrackForShape(shapeId, newTrackId)
    {
        $("#"+shapeId).attr("fill", CSS_COLOR_NAMES[newTrackId]);
        $("#"+shapeId).attr("data-track", newTrackId);
    }

    $("#save-btn").on("click", function(d) {
        onSaveClick();
    });

    $("#discard-btn").on("click", function(d) {
        onDiscardClick();
    });

});