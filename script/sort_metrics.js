sort_all = db.getCollection('runs')
    .find({ "info.overall_f1": { "$exists": true } })
    .sort({ "info.overall_f1": -1 }).limit(10)

sort_dev = db.getCollection('runs')
    .find({
        "info.overall_f1": { "$exists": true },
        "config.dev_corpus": /.*fold-i.*/,
        "config.train_corpus": /.*train-i.*/
    })
    .sort({ "info.overall_f1": -1 }).limit(25)

sort_test = db.getCollection('runs')
    .find({
        "command": "test",
        "info.overall_f1": { "$exists": true },
        "config.model_filename": /.*[dataset].*/
    })
    .sort({ "info.overall_f1": -1 }).limit(25)