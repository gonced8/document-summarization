from .retrosum import RetroSum


def get_model(args):
    name = args.model_name.lower()

    if "retrosum" in name:
        model = RetroSum
    else:
        print(f"Unrecognized model name: {name}")

    if args.from_checkpoint:
        print(f"Loading from checkpoint: {args.from_checkpoint}")
        model = model.load_from_checkpoint(args.from_checkpoint, **vars(args))
    else:
        model = model(args)

    return model
