// Creates canvas 320 × 200 at 10, 50
var paper = Raphael(0, 0, 4000, 3500);

function onNoteClick(e) {
    var trackId = e.target.dataset.track;
    var noteId  = e.target.id;
    var color   = CSS_COLOR_NAMES[trackId];

    $("#presentShapeValue").html(noteId);

    $("#presentTrackValue").html(trackId);
    $("#presentTrackColor").css("color", color);

    var trackUp   = trackId > 1           ? trackId-1 : trackId;
    var trackDown = trackId < maxTrackId  ? parseInt(trackId)+1 : trackId;

    $("#track1Value").html(trackUp);
    $("#track2Value").html(trackDown);

    $("#track1Color").css("color", CSS_COLOR_NAMES[trackUp]);
    $("#track2Color").css("color", CSS_COLOR_NAMES[trackDown]);

//    alert("Note ID: ".concat(e.target.id).concat("; Track ID: ").concat(e.target.dataset.track));
}

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

var selectionViewCoords = [];
var maxTrackId = 0;

fetch("data.json")
    .then(res => res.json())
    .then(data => {
        data["data"].forEach(shape => {
            var note = paper.path(shape.points)
                            .attr({"fill" : CSS_COLOR_NAMES[shape.track]});
            node_id = "shape_".concat(shape.id);
            note.node.id = node_id;
            note.node.dataset.track = shape.track;
            $("#".concat(node_id)).click(onNoteClick);
            $("#".concat(node_id)).on("mouseenter", hightlightNote);
            $("#".concat(node_id)).on("mouseout", lowlightNote);
            maxTrackId = Math.max(maxTrackId, shape.track);
        });
        selectionViewCoords = data["center"];

    $("body").append("<div id='selectionView' class='selectionView'></div>");
    $("#selectionView").css("top", selectionViewCoords[1]-selectionViewCoords[2]/2);
    $("#selectionView").css("left", selectionViewCoords[0]-selectionViewCoords[2]/2);
    $("#selectionView").css("width", selectionViewCoords[2]);
    $("#selectionView").css("height", selectionViewCoords[2]);
    
    $("#selectionView").append("<div id='selectionWrapper'></div>");
    $("#selectionWrapper").append("<div id='shapeWrapper'></div>");
    $("#shapeWrapper").append("<label id='presentShapeLabel' class='inline'>Shape-Id: </label>");
    $("#shapeWrapper").append("<div id='presentShapeValue' class='inline'></div>");

    
    $("#selectionWrapper").append("<div id='trackWrapper'></div>");
    $("#trackWrapper").append("<label id='presentTrackLabel' class='inline'>Spur-Id: </label>");
    $("#trackWrapper").append("<div id='presentTrackValue' class='inline'></div>");
    $("#trackWrapper").append("<label id='presentTrackColor' class='coloredSquare'> ◼</label>");

    $("#selectionView").append("<div id='selection1Wrapper' class='selection inline'></div>");
    $("#selectionView").append("<div id='selection2Wrapper' class='selection inline'></div>");

    
    $("#selection1Wrapper").append("<div id='track1Label' class='inline'>Tiefere Spur: </div>");
    $("#selection1Wrapper").append("<div id='track1Value' class='data inline'</div>");
    $("#selection1Wrapper").append("<div id='track1Color' class='inline'> ◼</div>");

    $("#selection2Wrapper").append("<div id='track2Label' class='inline'>Höhere Spur: </div>");
    $("#selection2Wrapper").append("<div id='track2Value' class='data inline'></div>");
    $("#selection2Wrapper").append("<div id='track2Color' class='inline'> ◼</div>");

    $("#selection1Wrapper").on("click", function(d) {
        
        var newTrack = $(this).children(".data").html();
        var shapeId  = $("#presentShapeValue").html();
        setNewTrackForShape(shapeId, newTrack);

    });
    $("#selection2Wrapper").on("click", function(d) {
        
        var newTrack = $(this).children(".data").html();
        var shapeId  = $("#presentShapeValue").html();
        setNewTrackForShape(shapeId, newTrack);

    });


    function setNewTrackForShape(shapeId, newTrackId)
    {
        $("#"+shapeId).attr("fill", CSS_COLOR_NAMES[newTrackId]);
        $("#"+shapeId).attr("data-track", newTrackId);
    }

});





